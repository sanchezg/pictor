#!/usr/bin/python

"""
    This is the main file for the pictor project.

    Pictor is a 'pictures interactions predictor':
    Given an input dataset, with about 50 different data from ~350000
    pictures from social networks, pictor tries to predict the interactions
    over any other picture, with that same info.

    Gonzalo Sanchez. 2016
"""

"""
    Execution requeriments:
        * Python 2.7
        * scikit-learn
        * NumPy
        * SciPy

    In order to execute pictor, just call this module as argument of the
    python interpreter:
        $ python pictor.py

    If no additional arguments are provided, pictor will lookup in the
    current folder the 'consolidated_features.csv' file with all the data.
    This file can be passed as parameter with the --csv flag:
        $ python pictor.py --csv='../consolidated_features.csv'

    This source code can be found in:
        https://github.com/sanchezg/pictor

    Tags:
    machine-learning, sklearn, python, prediction, pictures, interactions
"""

import sys
import argparse
from time import time

sys.path.append('tools/')

from format_tools import *
from regression_tools import *
from print_tools import *


def process_arguments():
    parser = argparse.ArgumentParser(
        description='Pictor, read https://github.com/sanchezg/pictor')
    parser.add_argument('--csv', type=str,
                        help='Route to input csv')
    args = parser.parse_args()
    return args


def get_filename():
    args = process_arguments()
    if args.csv is None:
        # Pick default location and filename
        csv_filename = '../consolidated_features.csv'
    else:
        csv_filename = args.csv
    return csv_filename


def load_and_format_data(filename):
    corpus_dataset, corpus_labels = load_dataset_from_csv(filename)
    features_undesired = load_features_from_file()
    discard_features(corpus_dataset, features_undesired)
    preformat_dataset(corpus_dataset)
    targets = split_dataset(corpus_dataset)
    # print corpus_dataset[0]
    # print targets[0]
    dataset = transform_dataset(corpus_dataset)
    # scale_dataset(dataset)
    X_train, X_test, y_train, y_test = conform_data(dataset, targets)
    make_prediction(X_train, y_train, X_test, y_test, show_score=True,
                    slice_samples=0)
    return


if __name__ == '__main__':
    csv_filename = get_filename()
    load_and_format_data(csv_filename)
