from __future__ import (
    print_function,
    division,
    absolute_import,
)

import logging

from deepbgcpred import util
from deepbgcpred.command.base import BaseCommand
import os
import shutil

from deepbgcpred.output.genbank import GenbankWriter
from deepbgcpred.output.pfam_tsv import PfamTSVWriter
from deepbgcpred.pipeline.annotator import DeepBGCpredAnnotator
from deepbgcpred.models.data_augmentation import DeepBGCpredAugmentation
from deepbgcpred.models.add_info import DeepBGCpredAddInfor


class PrepareCommand(BaseCommand):
    command = "prepare"
    help = """Prepare genomic sequence by annotating proteins and Pfam domains.
    
Examples:
    
  # Show detailed help 
  deepbgcpred prepare --help 
    
  # Detect proteins and pfam domains in a FASTA sequence and save the result as GenBank file 
  deepbgcpred prepare --pre --inputs sequence.fa --output-tsv sequence.prepared.tsv
  
  # 
  deepbgcpred prepare --add --clan-txt pfam34_clans.tsv --prepared-tsv sequence.prepared.tsv --output-new-tsv sequence.prepared.new.tsv
  
  # Data augmentation
  deepbgcpred prepare --aug --interaction-txt pfamA_interactions.txt --pfam-tsv sequence.prepared.new.tsv -r 0.02 -n 2 --output-aug-tsv sequence.prepared.aug.tsv
  """

    def add_arguments(self, parser):
        parser.add_argument(
            "--inputs",
            required=False,
            help="Input sequence file path(s) (FASTA/GenBank)",
        )
        parser.add_argument(
            "--limit-to-record",
            action="append",
            help="Process only specific record ID. Can be provided multiple times",
        )
        group = parser.add_argument_group("prepare arguments", "")
        group.add_argument("--pre", action="store_true", help="Ture for data prepare")
        group.add_argument(
            "--prodigal-meta-mode",
            action="store_true",
            default=False,
            help="Run Prodigal in '-p meta' mode to enable detecting genes in short contigs",
        )
        group.add_argument(
            "--protein",
            action="store_true",
            default=False,
            help="Accept amino-acid protein sequences as input (experimental). Will treat each file as a single record with multiple proteins.",
        )
        group.add_argument(
            "--output-gbk", required=False, help="Output GenBank file path"
        )
        group.add_argument("--output-tsv", required=False, help="Output TSV file path")

        group = parser.add_argument_group("data augmentation", "")
        group.add_argument(
            "--aug", action="store_true", help="Ture for data augmentation"
        )
        group.add_argument(
            "--interaction-txt", required=False, help="Pfam interaction TXT file"
        )
        group.add_argument("--pfam-tsv", required=False, help="Pfam TSV file")
        group.add_argument(
            "-r",
            "--replace-ratio",
            default=0.02,
            type=float,
            help="The replace ratio for each sequence",
        )
        group.add_argument(
            "-n",
            "--augment-number",
            default=2,
            type=int,
            help="The number of augmented samples for each sequence",
        )
        group.add_argument(
            "--output-aug-tsv", required=False, help="Output TSV file (augmented)"
        )

        group = parser.add_argument_group(
            "Add clan information to the prepared data", ""
        )
        group.add_argument(
            "--add",
            action="store_true",
            help="Ture for add clan information to the prepared data",
        )
        group.add_argument("--clan-txt", required=False, help="Pfam Clan TXT file")
        group.add_argument(
            "--prepared-tsv", required=False, help="Prepared TSV file path"
        )
        group.add_argument(
            "--output-new-tsv", required=False, help="Prepared TSV file path"
        )

    def run(
        self,
        inputs,
        limit_to_record,
        pre,
        output_gbk,
        output_tsv,
        prodigal_meta_mode,
        protein,
        aug,
        interaction_txt,
        pfam_tsv,
        replace_ratio,
        augment_number,
        output_aug_tsv,
        add,
        prepared_tsv,
        clan_txt,
        output_new_tsv,
    ):

        if pre:
            if not inputs:
                raise ValueError("Must input the sequence file path(s) (FASTA/GenBank)")

            first_output = output_gbk or output_tsv
            if not first_output:
                raise ValueError("Specify at least one of --output-gbk or --output-tsv")

            tmp_dir_path = first_output + ".tmp"
            logging.debug("Using TMP dir: %s", tmp_dir_path)
            if not os.path.exists(tmp_dir_path):
                os.mkdir(tmp_dir_path)

            prepare_step = DeepBGCpredAnnotator(
                tmp_dir_path=tmp_dir_path, prodigal_meta_mode=prodigal_meta_mode
            )

            writers = []
            if output_gbk:
                writers.append(GenbankWriter(out_path=output_gbk))
            if output_tsv:
                writers.append(PfamTSVWriter(out_path=output_tsv))

            num_records = 0
            for i, input_path in enumerate(inputs):
                logging.info(
                    "Processing input file %s/%s: %s", i + 1, len(inputs), input_path
                )
                with util.SequenceParser(input_path, protein=protein) as parser:
                    for record in parser.parse():
                        if limit_to_record and record.id not in limit_to_record:
                            logging.debug(
                                "Skipping record %s not matching filter %s",
                                record.id,
                                limit_to_record,
                            )
                            continue
                        prepare_step.run(record)
                        for writer in writers:
                            writer.write(record)
                        num_records += 1

            logging.debug("Removing TMP directory: %s", tmp_dir_path)
            shutil.rmtree(tmp_dir_path)

            prepare_step.print_summary()

            for writer in writers:
                writer.close()

            logging.info(
                "Saved %s fully annotated records to %s", num_records, first_output
            )
        elif aug:
            # Data Augmentation
            data_aug = DeepBGCpredAugmentation(
                interaction_txt, pfam_tsv, output_aug_tsv, replace_ratio, augment_number
            )
            data_aug.data_augmentation()

        elif add:
            # Add clan information and description to the prepared data
            data_add = DeepBGCpredAddInfor(clan_txt, prepared_tsv, output_new_tsv)
            data_add.add_information()
