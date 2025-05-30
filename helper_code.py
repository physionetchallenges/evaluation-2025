#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to use these functions in your code.

import hashlib
import numpy as np
import os
import scipy as sp
import sklearn
import sys
import wfdb

from collections import defaultdict

### Challenge variables
age_string = '# Age:'
sex_string = '# Sex:'
source_string = '# Source:'
label_string = '# Chagas label:'
probability_string = '# Chagas probability:'

### Challenge data I/O functions

# Find the records in a folder and its subfolders.
def find_records(folder, file_extension='.hea'):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == file_extension:
                record = os.path.relpath(os.path.join(root, file), folder)[:-len(file_extension)]
                records.add(record)
    records = sorted(records)
    return records

# Load the binary label for a record.
def load_label(record):
    header = load_header(record)
    label = get_label(header)
    return label

# Get the binary label from a WFDB header or a similar string.
def get_label(string, allow_missing=False):
    label, has_label = get_variable(string, label_string)
    if not has_label and not allow_missing:
        raise Exception('No label is available: are you trying to load the labels from the held-out data?')
    label = sanitize_boolean_value(label)
    return label

# Load the probability of a positive label for a record.
def load_probability(record):
    header = load_header(record)
    label = get_probability(header)
    return label

# Get the probability of a positive label from a WFDB header or a similar string.
def get_probability(string, allow_missing=False):
    probability, has_probability = get_variable(string, probability_string)
    if not has_probability and not allow_missing:
        raise Exception('No probability is available: are you trying to load the labels from the held-out data?')
    probability = sanitize_scalar_value(probability)
    return probability

# Save the model outputs for a record.
def save_outputs(output_file, record_name, label, probability):
    output_string = f'{record_name}\n{label_string} {label}\n{probability_string} {probability}\n'
    save_text(output_file, output_string)
    return output_string

# Load a text file as a string.
def load_text(filename):
    with open(filename, 'r') as f:
        string = f.read()
    return string

# Save a string as a text file.
def save_text(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

# Get a variable from a string.
def get_variable(string, variable_name):
    variable = None
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variable = l[len(variable_name):].strip()
            has_variable = True
    return variable, has_variable

# Load the age from a record.
def load_age(record):
    header = load_header(record)
    age = get_age(header)
    return age

# Get the age from a WFDB header or a similar string.
def get_age(string):
    age, has_age = get_variable(string, age_string)
    if is_number(age):
        age = float(age)
    return age

# Load the sex from a record.
def load_sex(record):
    header = load_header(record)
    sex = get_sex(header)
    return sex

# Get the sex from a WFDB header or a similar string.
def get_sex(string):
    sex, has_sex = get_variable(string, sex_string)
    return sex

# Load the sex from a record.
def load_source(record):
    header = load_header(record)
    source = get_source(header)
    return source

# Get the source from a WFDB header or a similar string.
def get_source(string):
    source, has_source = get_variable(string, source_string)
    return source

# Normalize the channel names.
def normalize_names(names_ref, names_est):
    tmp = list()
    for a in names_est:
        for b in names_ref:
            if a.casefold() == b.casefold():
                tmp.append(b)
                break
    return tmp

# Reorder channels in signal.
def reorder_signal(input_signal, input_channels, output_channels):
    # Do not allow repeated channels with potentially different values in a signal.
    assert(len(set(input_channels)) == len(input_channels))
    assert(len(set(output_channels)) == len(output_channels))

    if input_channels == output_channels:
        output_signal = input_signal
    else:
        output_channels = normalize_names(input_channels, output_channels)

        input_signal = np.asarray(input_signal)
        num_samples = np.shape(input_signal)[0]
        num_channels = len(output_channels)
        data_type = input_signal.dtype

        output_signal = np.zeros((num_samples, num_channels), dtype=data_type)
        for i, output_channel in enumerate(output_channels):
            for j, input_channel in enumerate(input_channels):
                if input_channel == output_channel:
                    output_signal[:, i] = input_signal[:, j]

    return output_signal

### WFDB functions

# Load the header for a record.
def load_header(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    return header

# Get the header file for a record.
def get_header_file(record):
    root, ext = os.path.splitext(record)
    if not ext == '.hea':
        header_file = record + '.hea'
    else:
        header_file = record
    return header_file

# Load the signals for a record.
def load_signals(record):
    signal, fields = wfdb.rdsamp(record)
    return signal, fields

# Get the signal files for a record.
def get_signal_files(record):
    header_file = get_header_file(record)
    header = load_header(record)
    path = os.path.dirname(record)

    signal_files = set()
    for i, l in enumerate(header.split('\n')):
        arrs = [arr.strip() for arr in l.split(' ')]
        if i == 0:
            num_signals = int(arrs[1])
        elif i <= num_signals:
            signal_file = os.path.join(path, arrs[0])
            signal_files.add(signal_file)
        else:
            break

    return signal_files

# Get the record name from a header file.
def get_record_name(string):
    value = string.split('\n')[0].split(' ')[0].split('/')[0].strip()
    return value

# Get the number of signals from a header file.
def get_num_signals(string):
    value = string.split('\n')[0].split(' ')[1].strip()
    if is_integer(value):
        value = int(value)
    else:
        value = None
    return value

# Get the sampling frequency from a header file.
def get_sampling_frequency(string):
    value = string.split('\n')[0].split(' ')[2].split('/')[0].strip()
    if is_number(value):
        value = float(value)
    else:
        value = None
    return value

# Get the number of samples from a header file.
def get_num_samples(string):
    value = string.split('\n')[0].split(' ')[3].strip()
    if is_integer(value):
        value = int(value)
    else:
        value = None
    return value

# Get the signal names from a header file.
def get_signal_names(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            value = l.split(' ')[8]
            values.append(value)
    return values

### Evaluation functions

# Compute the Challenge score.
def compute_challenge_score(labels, outputs, fraction_capacity = 0.05, num_permutations = 10**4, seed=12345):
    # Check the data.
    assert len(labels) == len(outputs)
    num_instances = len(labels)
    capacity = int(fraction_capacity * num_instances)

    # Convert the data to NumPy arrays, as needed, for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    # Permute the labels and outputs so that we can approximate the expected confusion matrix for "tied" probabilities.
    tp = np.zeros(num_permutations)
    fp = np.zeros(num_permutations)
    fn = np.zeros(num_permutations)
    tn = np.zeros(num_permutations)

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_permutations):
        permuted_idx = np.random.permutation(np.arange(num_instances))
        permuted_labels = labels[permuted_idx]
        permuted_outputs = outputs[permuted_idx]

        ordered_idx = np.argsort(permuted_outputs, stable=True)[::-1]
        ordered_labels = permuted_labels[ordered_idx]

        tp[i] = np.sum(ordered_labels[:capacity] == 1)
        fp[i] = np.sum(ordered_labels[:capacity] == 0)
        fn[i] = np.sum(ordered_labels[capacity:] == 1)
        tn[i] = np.sum(ordered_labels[capacity:] == 0)

    tp = np.mean(tp)
    fp = np.mean(fp)
    fn = np.mean(fn)
    tn = np.mean(tn)

    # Compute the true positive rate.
    if tp + fn > 0:
        tpr = tp / (tp + fn)
    else:
        tpr = float('nan')

    return tpr

def compute_auc(labels, outputs):
    import sklearn
    import sklearn.metrics

    auroc = sklearn.metrics.roc_auc_score(labels, outputs, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
    auprc = sklearn.metrics.average_precision_score(labels, outputs, average='macro', pos_label=1, sample_weight=None)

    return auroc, auprc

# Compute accuracy.
def compute_accuracy(labels, outputs):
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(labels, outputs, normalize=True, sample_weight=None)

    return accuracy

# Compute F-measure.
def compute_f_measure(labels, outputs):
    from sklearn.metrics import f1_score

    f_measure = f1_score(labels, outputs, labels=None, pos_label=1, average='weighted', sample_weight=None)

    return f_measure

### Other helper functions

# Remove any single or double quotes; parentheses, braces, and brackets (for singleton arrays); and spaces and tabs from a string.
def remove_extra_characters(x):
    x = str(x)
    x = x.replace('"', '').replace("'", "")
    x = x.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    x = x.replace(' ', '').replace('\t', '')
    x = x.strip()
    return x

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN, i.e., not a number, or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Check if a variable is a boolean or represents a boolean.
def is_boolean(x):
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x).casefold() in ('false', 'f', 'no', 'n')):
        return True
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x).casefold() in ('true', 't', 'yes', 'y')):
        return True
    else:
        return False

# Sanitize integer values.
def sanitize_integer_value(x):
    x = remove_extra_characters(x)
    if is_integer(x):
        return int(float(x))
    else:
        return float('nan')

# Sanitize scalar values.
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float('nan')

# Sanitize boolean values.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x).casefold() in ('false', 'f', 'no', 'n')):
        return 0
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x).casefold() in ('true', 't', 'yes', 'y')):
        return 1
    else:
        return float('nan')