import logging
from deepbgcpred.pipeline.pfam import HmmscanPfamRecordAnnotator
from deepbgcpred.pipeline.protein import ProdigalProteinRecordAnnotator
from deepbgcpred import util
from deepbgcpred.pipeline.step import PipelineStep
import os


class DeepBGCpredAnnotator(PipelineStep):
    def __init__(self, tmp_dir_path, prodigal_meta_mode=False):
        self.tmp_dir_path = tmp_dir_path
        self.prodigal_meta_mode = prodigal_meta_mode

    def run(self, record):
        util.fix_record_locus(record)
        util.fix_duplicate_cds(record)
        util.fix_dna_alphabet(record)

        record_tmp_path = os.path.join(
            self.tmp_dir_path, util.sanitize_filename(record.id)
        )
        logging.debug("Using record TMP prefix: %s", record_tmp_path)

        num_proteins = len(util.get_protein_features(record))
        if num_proteins:
            logging.info(
                "Sequence already contains %s CDS features, skipping CDS detection",
                num_proteins,
            )
        else:
            protein_annotator = ProdigalProteinRecordAnnotator(
                record=record,
                tmp_path_prefix=record_tmp_path,
                meta_mode=self.prodigal_meta_mode,
            )
            protein_annotator.annotate()

        num_pfams = len(util.get_pfam_features(record))
        if num_pfams:
            logging.info(
                "Sequence already contains %s Pfam features, skipping Pfam detection",
                num_pfams,
            )
        else:
            pfam_annotator = HmmscanPfamRecordAnnotator(
                record=record, tmp_path_prefix=record_tmp_path
            )
            pfam_annotator.annotate()

        util.sort_record_features(record)

    def print_summary(self):
        pass
