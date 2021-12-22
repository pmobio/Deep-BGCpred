import logging

from deepbgcpred.output.evaluation.pfam_score_plot import PfamScorePlotWriter
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

import pandas as pd

pfam_output = pd.DataFrame(columns=["True"])


class CurvePlotWriter(PfamScorePlotWriter):
    def __init__(self, out_path):
        super(CurvePlotWriter, self).__init__(out_path, max_sequences=None)

    @classmethod
    def get_description(cls):
        return "ROC curve based on predicted per-Pfam BGC scores"

    @classmethod
    def get_name(cls):
        return "roc-plot"

    def get_scores_and_responses(self):
        scores = []
        responses = []
        for i, (detector_scores, detector_names) in enumerate(
            zip(self.sequence_scores, self.sequence_detector_names)
        ):
            sample_responses = detector_scores["in_cluster"]
            responses.append(sample_responses)
            sample_scores = detector_scores.drop("in_cluster", axis=1)
            sample_scores.columns = detector_names
            scores.append(sample_scores)

        return scores, responses

    def plot_curve(
        self, true_values, predictions, ax=None, title=None, label=None, **kwargs
    ):
        raise NotImplementedError()

    def plot_extras(self, ax):
        pass

    def save_plot(self):
        scores, responses = self.get_scores_and_responses()

        if not scores:
            logging.debug(
                "No records were annotated, skipping evaluation plot %s", self.out_path
            )
            return

        merged_scores = pd.concat(scores, sort=False)
        merged_responses = pd.concat(responses, sort=False)

        if not merged_responses.sum():
            logging.debug(
                "No clusters were annotated, skipping evaluation plot %s", self.out_path
            )
            return

        logging.info(
            "%s: Plotting curve using %s samples", type(self).__name__, len(scores)
        )

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        self.plot_extras(ax)
        for name in merged_scores.columns:
            self.plot_curve(merged_responses, merged_scores[name], label=name, ax=ax)

        logging.debug("Saving evaluation curve plot to: %s", self.out_path)
        fig.savefig(self.out_path, dpi=150, bbox_inches="tight")


class ROCPlotWriter(CurvePlotWriter):
    @classmethod
    def get_description(cls):
        return "ROC curve based on predicted per-Pfam BGC scores"

    @classmethod
    def get_name(cls):
        return "roc-plot"

    def plot_extras(self, ax):
        ax.plot([0, 1], [0, 1], color="grey", lw=0.5, linestyle="--")

    def plot_curve(
        self,
        true_values,
        predictions,
        ax=None,
        title="ROC",
        label="ROC",
        lw=1,
        add_auc=True,
        **kwargs
    ):
        """
        Plot ROC curve of a single model. Can be called repeatedly with same axis to plot multiple curves.
        :param true_values: Series of true values
        :param predictions: Series of prediction values
        :param ax: Use given axis (will create new one if None)
        :param title: Plot title
        :param label: ROC curve label
        :param lw: Line width
        :param add_auc: Add AUC value to label
        :param baseline: Plot baseline that indicates performance of random model (AUC 0.5)
        :param figsize: Figure size
        :param kwargs: Additional arguments for plotting function
        :return: Figure axis
        """

        # record the pfam score to the file
        pfam_output["True"] = true_values
        pfam_output[label] = predictions
        output_path = "/".join(self.out_path.split(".")[0].split("/")[:-2])
        file_path = (
            output_path
            + "/"
            + self.out_path.split(".")[0].split("/")[-3]
            + ".pfam_output.tsv"
        )
        pfam_output.to_csv(file_path, index=False, sep="\t", encoding="utf-8")

        fpr, tpr, _ = roc_curve(true_values, predictions)
        roc_auc = auc(fpr, tpr)
        label_auc = label + ": {:.3f} AUC".format(roc_auc)
        logging.info("ROC result: %s", label_auc)
        ax.plot(fpr, tpr, lw=lw, label=label_auc if add_auc else label, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right", frameon=False)
        return ax
