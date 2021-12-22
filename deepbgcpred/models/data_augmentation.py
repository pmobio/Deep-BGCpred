#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ziyi Yang
# Data augmentation for BGC data

import pandas as pd
import random
from random import shuffle
import copy
import re


class DeepBGCpredAugmentation:
    def __init__(
        self, interaction_file, pfam_file, output_aug_tsv, replace_ratio, augment_num
    ):
        self.interaction_file = pd.read_table(interaction_file)
        self.pfam_file = pd.read_csv(pfam_file, sep="\t")
        self.replace_ratio = replace_ratio
        self.augment_num = augment_num
        self.output_aug_tsv = output_aug_tsv

    def data_augmentation(self):
        """
        Augment Data according to the original prepared training data
        :param interaction_file: Pfam interaction file path(s)
        :param pfam_file: the original prepared training data file path(s)
        :param output_aug_tsv: Pfam interaction file path(s)
        :param replace_ratio: the replaced ratio for sequence
        :param augment_num: the number of augmented samples for each sequence
        :return: Augmented Data
        """

        pfam_data = self.interaction_file
        org_data = self.pfam_file

        # constrcut a dictionary for synonyms of pfam domain
        sample_num = pfam_data.shape[0]
        pfam_dict = {}

        for i in range(sample_num):
            if pfam_data.loc[i, pfam_data.columns[0]] in pfam_dict.keys():
                key = pfam_data.loc[i, pfam_data.columns[0]]
                value = pfam_data.loc[i, pfam_data.columns[1]]
                pfam_dict[key].append(value)
            else:
                key = pfam_data.loc[i, pfam_data.columns[0]]
                value = pfam_data.loc[i, pfam_data.columns[1]]
                pfam_dict[key] = [value]

        # Integrate all input data
        all_samples = []
        samples = [i for i in org_data.groupby("sequence_id")]
        all_samples += samples

        # Data augmentation
        alpha_sr = self.replace_ratio
        num_aug = self.augment_num
        sample_augment_list = []

        for i in range(len(all_samples)):
            sample_id = all_samples[i][0]
            sample_info = all_samples[i][1]
            pfam_str = " "
            pfam_str_list = all_samples[i][1].loc[:, "pfam_id"].values.tolist()
            pfam_str_list = [
                str(i) for i in pfam_str_list
            ]  # list has nan with float type
            sentence = pfam_str.join(pfam_str_list)

            # Data Augmentation
            aug_sentences = eda(
                sentence, dict=pfam_dict, alpha_sr=alpha_sr, num_aug=num_aug
            )

            # assign the generated data to the original data
            for j in range(len(aug_sentences)):
                new_samples = all_samples[i][1]

                aug_sentence_list = aug_sentences[j].split()
                new_samples["pfam_id"] = aug_sentence_list

                sample_name = sample_id.split(".")[0]
                sample_idx = sample_id.split(".")[1]
                sample_new_id = sample_name + "_" + str(j) + "." + sample_idx
                new_samples["sequence_id"] = [sample_new_id] * sample_info.shape[0]

                new_samples_copy = copy.deepcopy(new_samples)
                sample_augment_list.append(new_samples_copy)

        sample_augment = pd.concat(sample_augment_list, sort=False)
        sample_augment = sample_augment.reset_index(drop=True)

        # save the data to the file
        print("Save the augmentation file...")
        sample_augment.to_csv(
            self.output_aug_tsv, index=False, sep="\t", encoding="utf-8"
        )


def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")

    for char in line:
        if char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ":
            clean_line += char
        else:
            clean_line += " "

    clean_line = re.sub(" +", " ", clean_line)  # delete extra spaces
    if clean_line[0] == " ":
        clean_line = clean_line[1:]
    return clean_line


def synonym_replacement(words, n, dict):
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word, dict)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = " ".join(new_words)
    new_words = sentence.split(" ")

    return new_words


def get_synonyms(word, dict):
    synonyms = set()
    pfam_dict = dict
    # print(word)
    if word in pfam_dict.keys():
        synonyms = pfam_dict[word]
    return synonyms


def eda(sentence, dict, alpha_sr=0.1, num_aug=2):
    sentence = get_only_chars(sentence)
    words = sentence.split(" ")
    words = [word for word in words if word is not ""]
    num_words = len(words)

    augmented_sentences = []

    if alpha_sr > 0:
        n_sr = max(2, int(alpha_sr * num_words))
        for _ in range(num_aug):
            a_words = synonym_replacement(words, n_sr, dict)
            augmented_sentences.append(" ".join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [
            s for s in augmented_sentences if random.uniform(0, 1) < keep_prob
        ]

    # append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences
