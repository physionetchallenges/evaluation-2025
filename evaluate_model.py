#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d data -o outputs -s scores.csv
#
# where 'data' is a folder containing WFDB files with the labels for the data, 'outputs' is a folder containing WFDB files with
# outputs from your model, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each data or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
# described on the Challenge webpage.

import argparse
import numpy as np
import os
import os.path
import pandas as pd
import sys

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Evaluate the models.
def evaluate_model(data_folder, output_folder):
    # Load the labels from the data.
    record_to_label = dict()
    if os.path.isdir(data_folder):
        records = find_records(data_folder)
        for i, record in enumerate(records):
            label_filename = os.path.join(data_folder, record) + '.hea'
            text = load_text(label_filename)
            label = get_label(text, allow_missing=False)
            record_to_label[record] = label
    else:
        NotImplementedError(f'{data_folder} is not a directory or folder.')

    if not record_to_label:
        raise FileNotFoundError('No data labels found.')

    # Load the labels from the outputs.
    record_to_binary_output = dict()
    record_to_probability_output = dict()
    if os.path.isdir(output_folder):
        records = find_records(output_folder, file_extension='.txt')
        for i, record in enumerate(records):
            output_filename = os.path.join(output_folder, record) + '.txt'
            text = load_text(output_filename)
            binary_output = get_label(text, allow_missing=True)
            probability_output = get_probability(text, allow_missing=True)
            record_to_binary_output[record] = binary_output
            record_to_probability_output[record] = probability_output
    else:
        NotImplementedError(f'File I/O for {output_folder} not implemented')

    if not record_to_binary_output:
        raise FileNotFoundError('No binary outputs found.')

    if not record_to_probability_output:
        raise FileNotFoundError('No probability outputs found.')

    # Convert the labels and outputs to vectors.
    records = sorted(record_to_label)
    num_records = len(records)

    labels = np.zeros(num_records)
    binary_outputs = np.zeros(num_records)
    probability_outputs = np.zeros(num_records)

    for i, record in enumerate(records):
        label = record_to_label[record]
        labels[i] = label

        # Replace missing binary outputs with zeros so that we can evaluate an output for each record.
        if record in record_to_binary_output:
            binary_output = record_to_binary_output[record]
        else:
            binary_output = float('nan')
        if not is_nan(binary_output):
            binary_outputs[i] = binary_output
        else:
            binary_outputs[i] = 0

        # Replace missing probability outputs with zeros so that we can evaluate an output for each record.
        if record in record_to_probability_output:
            probability_output = record_to_probability_output[record]
        else:
            probability_output = float('nan')
        if not is_nan(probability_output):
            probability_outputs[i] = probability_output
        else:
            probability_outputs[i] = 0

    # Evaluate the model outputs.
    challenge_score = compute_challenge_score(labels, probability_outputs)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)

    return challenge_score, auroc, auprc, accuracy, f_measure

# Run the code.
def run(args):
    # Compute the scores for the model outputs.
    challenge_score, auroc, auprc, accuracy, f_measure = evaluate_model(args.data_folder, args.output_folder)

    output_string = \
        f'Challenge score: {challenge_score:.3f}\n' + \
        f'AUROC: {auroc:.3f}\n' \
        f'AUPRC: {auprc:.3f}\n' + \
        f'Accuracy: {accuracy:.3f}\n' \
        f'F-measure: {f_measure:.3f}\n'

    # Output the scores to screen and/or a file.
    if args.score_file:
        save_text(args.score_file, output_string)
    else:
        print(output_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))