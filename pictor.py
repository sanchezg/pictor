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
from print_tools import *
from regression_tools import make_prediction

# Used to print duration of each function
show_verbose = True

def process_arguments():
    parser = argparse.ArgumentParser(
        description='Pictor. More info: https://github.com/sanchezg/pictor')
    parser.add_argument('--csv', type=str,
                        help='Path to input csv with dataset')
    parser.add_argument('--feat', type=str,
                        help='Path to file with features to discard')
    parser.add_argument('--verbose', type=str,
                        help='"y" if you want to show verbose output (default). "n" otherwise.')
    args = parser.parse_args()
    return args


def get_args_parsed():
    args = process_arguments()
    # Default input values
    csv_filename = 'consolidated_features.csv'
    feat_filename = ''
    if args.csv is not None:
        csv_filename = args.csv
    if args.feat is not None:
        feat_filename = args.feat
    if args.verbose is not None:
        if args.verbose == "y":
            show_verbose = True
        elif args.verbose == "n":
            show_verbose = False
        else:
            print "Incorrect verbose arg. Assumed 'y'es."
    return csv_filename, feat_filename


def load_and_format_data(dataset_filename, discard_feat_filename):
    corpus_dataset, corpus_labels = load_dataset_from_csv(dataset_filename,
                                                          sv=show_verbose)
    if corpus_dataset == []:
        return -1
    if discard_feat_filename != '':
        features_undesired = load_features_from_file(discard_feat_filename)
        discard_features(corpus_dataset, features_undesired, sv=show_verbose)
    preformat_dataset(corpus_dataset, sv=show_verbose)
    targets = split_dataset(corpus_dataset, sv=show_verbose)
    # print corpus_dataset[0]
    dataset = transform_dataset(corpus_dataset, sv=show_verbose)
    del corpus_dataset  # Free some memory
    # scale_dataset(dataset)
    X_train, X_test, y_train, y_test = conform_data(dataset, targets,
                                                    sv=show_verbose)
    del dataset  # Free some memory
    del targets  # Free some memory
    make_prediction(X_train, y_train, X_test, y_test, sv=show_verbose,
                    slice_samples=0)
    return 0


if __name__ == '__main__':
    print "Welcome to pictor: a predictor for pictures interactions"
    csv_filename, features_filename = get_args_parsed()
    load_and_format_data(csv_filename, features_filename)
