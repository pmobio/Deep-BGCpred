#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ziyi Yang
# Add Clan and description information to the prepared data

import pandas as pd


class DeepBGCpredAddInfor:
    def __init__(self, clan_txt, prepared_tsv, output_new_tsv):
        self.pfam_clan = pd.read_table(clan_txt)
        self.data = pd.read_csv(prepared_tsv, sep="\t")
        self.output_file = output_new_tsv

    def add_information(self):
        """
        Augment Data according to the original prepared training data
        :param interaction_txt: Pfam interaction file path(s)
        :param prepared_tsv: the original prepared training data file path(s)
        :param output_new_tsv: data with Pfam, Clan and description file path(s)
        :return: Data with Pfam, Clan and description information
        """
        pfam_clan_dict = {}
        pfam_desc_dict = {}

        print(self.pfam_clan.columns)
        print(self.data.columns)

        for i in range(self.pfam_clan.shape[0]):
            pfam_clan_dict[self.pfam_clan.iloc[i]["pfamA_acc"]] = self.pfam_clan.iloc[
                i
            ]["clan_acc"]
            pfam_desc_dict[self.pfam_clan.iloc[i]["pfamA_acc"]] = self.pfam_clan.iloc[
                i
            ]["description"]

        # Add clan id and description to the prepared file
        self.data["clan_id"] = self.data["pfam_id"].map(pfam_clan_dict)
        self.data["description"] = self.data["pfam_id"].map(pfam_desc_dict)

        # Add NULL to the NA column
        self.data["clan_id"].fillna("NULL", inplace=True)
        self.data["description"].fillna("NULL", inplace=True)

        # Save the file
        print("Save the file after add the clan and description information...")
        self.data.to_csv(self.output_file, index=None, sep="\t")
