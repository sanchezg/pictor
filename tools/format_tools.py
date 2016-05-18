#!/usr/bin/python

import numpy
from time import time

"""This file provides functions and tools for formatting data into different
options.
"""


def load_dataset_from_csv(filename, delimiter='|', first_line=True,
                          output_format='dict', sv=True):
    """This function tries to load a list of dictionaries from a csv file.
    Parameter 'first_line' indicates if the first line in the csv file
    contains the labels for the dataset. These labels are used as keys
    in the dictionary.
    The function returns the dataset (as a list of dict) and the labels
    (as a list of str).
    """
    if sv:
        print 'Loading dataset from csv file...'
        t0 = time()

    labels = []
    dataset = []
    first_line_loaded = False
    try:
        file_obj = open(filename, 'r')
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        return [], []

    # CSV data will be loaded as:
    # 'key': {
    #        'ikey1': value,
    #         'ikey2': value,
    #        ...
    #        'ikeyn': value
    #        }

    lines = file_obj.readlines()
    file_obj.close()

    for line in lines:
        data_line = line.split(delimiter)
        if first_line_loaded:
            inner_dict = dict(zip(labels, data_line))
            dataset.append(inner_dict)
        else:
            labels = data_line
            first_line_loaded = True
    if sv:
        print "Time loading dataset: {}".format(time() - t0)
    return dataset, labels


def load_features_from_file(filename):
    """This function loads a list with features names from the passed
    argument.
    """
    features_list = []
    try:
        f = open(filename, 'r')
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        return []

    lines = f.readlines()
    f.close()
    features_list = [str(feat).split('\n')[0] for feat in lines
                     if not feat.startswith('#')]

    if 'vertical' in features_list:
        idx = features_list.index('vertical')
        features_list[idx] = 'vertical\n'
    return features_list


def discard_features(dataset_d, features_unwanted, sv=True):
    """This function removes from dataset_d all features which labels are in
    features_unwanted.
    dataset_d should be a list of dicts as the one returned from
    'load_dataset_from_csv'.
    features_unwanted should be a list of labels in str format.
    """
    if sv:
        print 'Discarding undesired features...'
        t0 = time()

    for row in dataset_d:
        for feature in features_unwanted:
            try:
                del row[feature]
            except KeyError:
                pass
    if sv:
        print "Time discarding features: {}".format(time() - t0)
    return


def preformat_dataset(dataset_d, sv=True):
    """This function takes the dataset loaded with 'load_dataset_from_csv'
    function and converts all numerical values to float type.
    """
    if sv:
        print 'Preformatting dataset...'
        t0 = time()

    count = len(dataset_d)
    labels = dataset_d[0].keys()
    for idx in xrange(count):
        for label in labels:
            dataset_d[idx][label] = autoformat_element(dataset_d[idx][label])
    if sv:
        print "Time preformatting dataset: {}".format(time() - t0)
    return


def scale_dataset(corpus_dataset):
    """Uses sklearn.preprocessing module to scale features values.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    corpus_dataset = scaler.fit_transform(corpus_dataset)
    return


def split_dataset(dataset_d, target_feature='interactions', sv=True):
    """This function removes from the input dataset those values corresponding
    to the target feature (output), and append them to other list returned at
    the end.
    """
    if sv:
        print 'Splitting targets...'
        t0 = time()

    output_values = []
    for row in dataset_d:
        output_values.append(row.pop(target_feature))

    if sv:
        print "Time splitting dataset: {}".format(time() - t0)
    return output_values


def transform_dataset(dataset_d, sv=True):
    """Uses sklearn DictVectorizer to transform the dataset and convert inner
    categorical features in a suitable representation.
    """
    from sklearn.feature_extraction import DictVectorizer

    if sv:
        print 'Transforming dataset...'
        t0 = time()
    vec = DictVectorizer(sparse=False)
    dataset_t = vec.fit_transform(dataset_d)

    if sv:
        print "Time transforming dataset: {}".format(time() - t0)
    return dataset_t


def conform_data(dataset, targets, test_prop=0.30, sv=True):
    """This function takes the dataset returned by 'transform_dataset'
    function and returns the trainer and tests arrays returned by
    sklearn.cross_validation module.
    """
    from sklearn.cross_validation import train_test_split

    if sv:
        print 'Conforming training and testing arrays...'
        t0 = time()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, targets, test_size=test_prop, random_state=42)

    if sv:
        print "Time conforming data: {}".format(time() - t0)
    return X_train, X_test, y_train, y_test


def autoformat_element(element):
    """Tries to convert the passed parameter in float format.
    If the input parameter is of str type then only a sanitization of str is
    performed.
    """
    try:
        # Some elements can be empty, they should contain a value
        if element == '':
            element_formatted = 0.
        else:
            element_formatted = float(element)
    except ValueError:
        # The element is of str type
        element_formatted = str_sanitization(element)
    return element_formatted


def str_sanitization(element):
    """Erases the '\n' at the end in some str objects"""
    return element.split('\n')[0]
