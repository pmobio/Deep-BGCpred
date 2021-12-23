#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ziyi Yang
# The dual-model serial screening

import pandas as pd
import os

class DeepBGCpredScreening():

    def __init__(
        self, output, detectors, classifiers
    ):
        self.output= output
        self.detectors = detectors
        self.classifiers = classifiers

    def screening(self):
        """
        The dual-model serial screening
        :param output: Results save path(s)
        :param detectors: list of detectors
        :param classifiers: list of classifiers
        :return: pfam and BGCs results with the dual-model serial screening
        """

        # check each detector and classifier for the dual-model serial screening.

        if not os.path.exists(os.path.join(self.output, "screen")):
            os.mkdir(os.path.join(self.output, "screen"))

        for detector, classifier in zip(self.detectors, self.classifiers):
            # load the pfam and class file
            prefix_name = self.output.split("/")[-1]
            pfam_bgc = pd.read_csv(
                os.path.join(self.output, prefix_name + ".pfam.tsv"), sep="\t"
            )
            class_bgc = pd.read_csv(
                os.path.join(self.output, prefix_name + ".bgc.tsv"), sep="\t"
            )
            pfam_old = pd.read_csv(
                os.path.join(self.output, prefix_name + ".pfam_output.tsv"), sep="\t"
            )

            # select detector information from the full table
            detector = detector.split("/")[-1].split(".")[0]
            classifier = classifier.split("/")[-1].split(".")[0]
            model_bgc = class_bgc.loc[
                class_bgc["detector"] == detector
                ].reset_index(drop=True)
            model_sub_bgc = model_bgc.loc[
                model_bgc[classifier] == "Non_BGC"
                ].reset_index(drop=True)
            class_sub_bgc = class_bgc.loc[
                class_bgc[detector + "_score"] == "Non_BGC"
                ].reset_index(drop=True)

            model_concat_bgc = pd.concat(
                [model_sub_bgc, class_sub_bgc]
            ).reset_index(drop=True)

            for i in range(model_concat_bgc.shape[0]):
                nucl_start = model_concat_bgc["nucl_start"][i]
                nucl_end = model_concat_bgc["nucl_end"][i]
                sequence_id = model_concat_bgc["sequence_id"][i]
                pfam_sub_bgc = pfam_bgc[
                    (pfam_bgc["gene_start"] >= nucl_start)
                    & (pfam_bgc["gene_end"] <= nucl_end)
                    ]
                pfam_sub_idx = pfam_sub_bgc[
                    pfam_sub_bgc["sequence_id"] == sequence_id
                    ].index
                pfam_bgc.loc[pfam_sub_idx, detector + "_score"] = 0

            pfam_old = pfam_old.drop([detector], axis=1)
            pfam_new = pd.concat([pfam_old, pfam_bgc[detector + "_score"]], axis=1)
            pfam_new = pfam_new.rename(columns={detector + "_score": detector})

            class_bgc_s = class_bgc[class_bgc["detector"] == detector].reset_index(
                drop=True
            )
            class_bgc_s = class_bgc_s.drop(
                class_bgc_s[class_bgc_s[classifier] == "Non_BGC"].index
            )

            pfam_new.to_csv(
                os.path.join(
                    self.output,
                    "Screen/" + detector + "_" + classifier + ".pfam_filter.tsv",
                ),
                index=None,
                sep="\t",
            )
            class_bgc_s.to_csv(
                os.path.join(
                    self.output,
                    "Screen/" + detector + "_" + classifier + ".bgc_filter.tsv",
                ),
                index=None,
                sep="\t",
            )