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
"""

import sys
from time import time

sys.path.append('tools/')

import format_tools
import regression_tools
from print_tools import print_some_data, print_from_dataset, inspect_dataset

