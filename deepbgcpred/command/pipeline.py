from __future__ import (
    print_function,
    division,
    absolute_import,
)

import logging

import deepbgcpred.util
from deepbgcpred.command.base import BaseCommand
import os
from deepbgcpred import util
from Bio import SeqIO

from deepbgcpred.output.bgc_genbank import BGCGenbankWriter
from deepbgcpred.output.evaluation.pr_plot import PrecisionRecallPlotWriter
from deepbgcpred.output.evaluation.roc_plot import ROCPlotWriter
from deepbgcpred.output.readme import ReadmeWriter
from deepbgcpred.pipeline.annotator import DeepBGCpredAnnotator
from deepbgcpred.pipeline.detector import DeepBGCpredDetector
from deepbgcpred.pipeline.classifier import DeepBGCpredClassifier
from deepbgcpred.output.genbank import GenbankWriter
from deepbgcpred.output.evaluation.bgc_region_plot import BGCRegionPlotWriter
from deepbgcpred.output.cluster_tsv import ClusterTSVWriter
from deepbgcpred.output.evaluation.pfam_score_plot import PfamScorePlotWriter
from deepbgcpred.output.pfam_tsv import PfamTSVWriter
from deepbgcpred.output.antismash_json import AntismashJSONWriter


class PipelineCommand(BaseCommand):
    command = 'pipeline'

    help = """Run DeepBGCpred pipeline: Preparation, BGC detection, BGC classification and generate the report directory.
    
Examples:
    
  # Show detailed help 
  deepbgcpred pipeline --help 
    
  # Detect BGCs in a nucleotide FASTA sequence using DeepBGCpred model 
  deepbgcpred pipeline sequence.fa
  
  # Detect and classify BGCs in mySequence.fa using the DeepBGCpred detector without sliding window strategy
  python main.py pipeline mySequence.fa --pcfile --pfam-clain-file pfam34_clans.tsv --detector myDetector.pkl --classifier myClassifier.pkl
  
  # Detect and classify BGCs in mySequence.fa using the DeepBGCpred detector with sliding window strategy
  python main.py pipeline mySequence.fa --pcfile --pfam-clain-file pfam34_clans.tsv --detector myDetector.pkl --classifier myClassifier.pkl --sliding-window -sw_width 256 -sw_steps 20
  
  # Detect BGCs using the ClusterFinder GeneBorder detection model and a higher score threshold
  deepbgcpred pipeline --detector clusterfinder_geneborder --score 0.8 sequence.fa
  
  # Add additional clusters detected using DeepBGCpred model with a strict score threshold
  deepbgcpred pipeline --continue --output sequence/ --label deepbgcpred --score 0.9 sequence/sequence.full.gbk
  """

    LOG_FILENAME = 'LOG.txt'
    PLOT_DIRNAME = 'evaluation'
    TMP_DIRNAME = 'tmp'

    def add_arguments(self, parser):

        parser.add_argument(dest='inputs', nargs='+', help="Input sequence file path (FASTA, GenBank, Pfam CSV)")

        parser.add_argument('-o', '--output', required=False, help="Custom output directory path")
        parser.add_argument('--limit-to-record', action='append', help="Process only specific record ID. Can be provided multiple times")
        parser.add_argument('--minimal-output', dest='is_minimal_output', action='store_true', default=False,
                            help="Produce minimal output with just the GenBank sequence file")
        parser.add_argument('--prodigal-meta-mode', action='store_true', default=False, help="Run Prodigal in '-p meta' mode to enable detecting genes in short contigs")
        parser.add_argument('--protein', action='store_true', default=False, help="Accept amino-acid protein sequences as input (experimental). Will treat each file as a single record with multiple proteins.")
        parser.add_argument('--pfam-clain-file', help='Pfam clain annotation information file.')
        parser.add_argument('--pcfile', action='store_true', help='Accept Pfam clain annotation information file as input.')

        group = parser.add_argument_group('BGC detection options', '')
        no_models_message = 'run "deepbgcpred download" to download models'
        detector_names = util.get_available_models('detector')
        group.add_argument('-d', '--detector', dest='detectors', action='append', default=[],
                           help="Trained detection model name ({}) or path to trained model pickle file. "
                                "Can be provided multiple times (-d first -d second)".format(', '.join(detector_names) or no_models_message))
        group.add_argument('--no-detector', action='store_true', help="Disable BGC detection")
        group.add_argument('-l', '--label', dest='labels', action='append', default=[], help="Label for detected clusters (equal to --detector by default). "
                                                                                             "If multiple detectors are provided, a label should be provided for each one")
        group.add_argument('-s', '--score', default=0.5, type=float,
                            help="Average protein-wise DeepBGCpred score threshold for extracting BGC regions from Pfam sequences (default: %(default)s)")
        group.add_argument('-w', '--sliding-window', action='store_true', help='Adopt sliding window strategy')
        group.add_argument('-sw_width', type=int, default=256, help='width of sliding window')
        group.add_argument('-sw_steps', type=int, default=20, help='step length of sliding window')
        group.add_argument('-i', '--input-size', nargs='+', type=int, default=[102, 64, 64],
                            help="list type, [102, 64, 64] for Deep-BGCpred, [102] for DeepBGC")
        group.add_argument('--merge-max-protein-gap', default=0, type=int, help="Merge detected BGCs within given number of proteins (default: %(default)s)")
        group.add_argument('--merge-max-nucl-gap', default=0, type=int, help="Merge detected BGCs within given number of nucleotides (default: %(default)s)")
        group.add_argument('--min-nucl', default=1, type=int, help="Minimum BGC nucleotide length (default: %(default)s)")
        group.add_argument('--min-proteins', default=1, type=int, help="Minimum number of proteins in a BGC (default: %(default)s)")
        group.add_argument('--min-domains', default=1, type=int, help="Minimum number of protein domains in a BGC (default: %(default)s)")
        group.add_argument('--min-bio-domains', default=0, type=int, help="Minimum number of known biosynthetic (as defined by antiSMASH) protein domains in a BGC (default: %(default)s)")

        group = parser.add_argument_group('BGC classification options', '')
        classifier_names = util.get_available_models('classifier')
        group.add_argument('-c', '--classifier', dest='classifiers', action='append', default=[],
                            help="Trained classification model name ({}) or path to trained model pickle file. "
                                 "Can be provided multiple times (-c first -c second)".format(', '.join(classifier_names) or no_models_message))
        group.add_argument('--no-classifier', action='store_true', help="Disable BGC classification")
        group.add_argument('--classifier-score', default=0.5, type=float,
                            help="DeepBGCpred classification score threshold for assigning classes to BGCs (default: %(default)s)")

    def run(self, inputs, output, detectors, no_detector, labels, classifiers, no_classifier,
            is_minimal_output, limit_to_record, score, classifier_score, merge_max_protein_gap, merge_max_nucl_gap, min_nucl,
            min_proteins, min_domains, min_bio_domains, prodigal_meta_mode, protein, pfam_clain_file, input_size, sliding_window,
            sw_width, sw_steps, pcfile):
        if not detectors:
            detectors = ['deepbgcpred']

        if pcfile:
            if not pfam_clain_file:
                raise ValueError('Please input the Pfam Clain annotation file...')

        if not classifiers:
            classifiers = ['product_class', 'product_activity']
        if not output:
            # if not specified, set output path to name of first input file without extension
            output, _ = os.path.splitext(os.path.basename(os.path.normpath(inputs[0])))

        if not os.path.exists(output):
            os.mkdir(output)

        # Save log to LOG.txt file
        logger = logging.getLogger('')
        logger.addHandler(logging.FileHandler(os.path.join(output, self.LOG_FILENAME)))

        # Define report dir paths
        tmp_path = os.path.join(output, self.TMP_DIRNAME)
        evaluation_path = os.path.join(output, self.PLOT_DIRNAME)
        output_file_name = os.path.basename(os.path.normpath(output))

        steps = []
        steps.append(DeepBGCpredAnnotator(tmp_dir_path=tmp_path, prodigal_meta_mode=prodigal_meta_mode))

        if not no_detector:
            if not labels:
                labels = [None] * len(detectors)
            elif len(labels) != len(detectors):
                raise ValueError('A separate label should be provided for each of the detectors: {}'.format(detectors))

            for detector, label in zip(detectors, labels):
                steps.append(DeepBGCpredDetector(
                    detector=detector,
                    pfamfile=pfam_clain_file,
                    input_size=input_size,
                    sliding_window=sliding_window,
                    sw_width=sw_width,
                    sw_steps=sw_steps,
                    label=label,
                    pcfile=pcfile,
                    score_threshold=score,
                    merge_max_protein_gap=merge_max_protein_gap,
                    merge_max_nucl_gap=merge_max_nucl_gap,
                    min_nucl=min_nucl,
                    min_proteins=min_proteins,
                    min_domains=min_domains,
                    min_bio_domains=min_bio_domains
                ))

        writers = []
        writers.append(GenbankWriter(out_path=os.path.join(output, output_file_name+'.full.gbk')))
        writers.append(AntismashJSONWriter(out_path=os.path.join(output, output_file_name + '.antismash.json')))
        is_evaluation = False
        if not is_minimal_output:
            writers.append(BGCGenbankWriter(out_path=os.path.join(output, output_file_name+'.bgc.gbk')))
            writers.append(ClusterTSVWriter(out_path=os.path.join(output, output_file_name+'.bgc.tsv')))
            writers.append(PfamTSVWriter(out_path=os.path.join(output, output_file_name+'.pfam.tsv')))

            is_evaluation = True
            writers.append(PfamScorePlotWriter(out_path=os.path.join(evaluation_path, output_file_name + '.score.png')))
            writers.append(BGCRegionPlotWriter(out_path=os.path.join(evaluation_path, output_file_name+'.bgc.png')))
            writers.append(ROCPlotWriter(out_path=os.path.join(evaluation_path, output_file_name+'.roc.png')))
            writers.append(PrecisionRecallPlotWriter(out_path=os.path.join(evaluation_path, output_file_name+'.pr.png')))

        writers.append(ReadmeWriter(out_path=os.path.join(output, 'README.txt'), root_path=output, writers=writers))

        if not no_classifier:
            for classifier in classifiers:
                steps.append(DeepBGCpredClassifier(classifier=classifier, score_threshold=classifier_score))

        # Create temp and evaluation dir
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        if is_evaluation:
            if not os.path.exists(evaluation_path):
                os.mkdir(evaluation_path)

        record_idx = 0
        for i, input_path in enumerate(inputs):
            logging.info('Processing input file %s/%s: %s', i+1, len(inputs), input_path)
            with util.SequenceParser(input_path, protein=protein) as parser:
                for record in parser.parse():
                    if limit_to_record and record.id not in limit_to_record:
                        logging.debug('Skipping record %s not matching filter %s', record.id, limit_to_record)
                        continue

                    record_idx += 1
                    logging.info('='*80)
                    logging.info('Processing record #%s: %s', record_idx, record.id)
                    for step in steps:
                        step.run(record)

                    logging.info('Saving processed record %s', record.id)
                    for writer in writers:
                        writer.write(record)

        logging.info('=' * 80)
        for step in steps:
            step.print_summary()

        for writer in writers:
            writer.close()

        logging.info('='*80)
        logging.info('Saved DeepBGCpred result to: {}'.format(output))
