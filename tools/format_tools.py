#!/usr/bin/python

import numpy
from time import time

"""
This file provides functions and tools for formatting data into different
options.
"""


def load_dataset_from_csv(filename, delimiter='|', features_to_discard = []):
    """Tries to load the dataset from a csv file.
    The function returns the dataset loaded as a list of dicts and the labels
    as a list of str."""

    labels = []
    dataset = []
    first_line_loaded = False
    try:
        file_obj = open(filename, 'r')
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        return [], []

    lines = file_obj.readlines()
    file_obj.close()

    for line in lines:
        data_line = map(autoformat_element, line.split(delimiter))
        if first_line_loaded:
            inner_dict = dict(zip(labels, data_line))
            dataset.append(inner_dict)
        else:
            labels = data_line
            first_line_loaded = True

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

    # This label has a different form
    if 'vertical' in features_list:
        idx = features_list.index('vertical')
        features_list[idx] = 'vertical\n'
    return features_list


def discard_features(dataset_d, features_unwanted):
    """Removes from dataset_d those features which labels are in
    features_unwanted.
    dataset_d should be a list of dicts as the one returned from
    'load_dataset_from_csv'.
    features_unwanted should be a list of labels in str format."""
    for row in dataset_d:
        for feature in features_unwanted:
            try:
                del row[feature]
            except KeyError:
                pass
    return


def clean_dataset(dataset, labels_values):
    """
    """
    count = 0
    for sample in dataset:
        for feature, value in labels_values:
            try:
                if sample[feature] == value:
                    del sample[feature]
                    count += 1
            except KeyError:
                pass
    return count


def scale_dataset(corpus_dataset):
    """Uses sklearn.preprocessing module to scale features values.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    corpus_dataset = scaler.fit_transform(corpus_dataset)
    return


def split_dataset(dataset_d, target_feature='interactions'):
    """This function removes from the input dataset those values corresponding
    to the target feature (output), and append them to other list returned at
    the end.
    """
    output_values = []
    for row in dataset_d:
        output_values.append(row.pop(target_feature))

    return output_values


def transform_dataset(dataset_d):
    """Uses sklearn DictVectorizer to transform the dataset and convert inner
    categorical features in a suitable representation.
    """
    from sklearn.feature_extraction import DictVectorizer

    vec = DictVectorizer(sparse=False)
    dataset_t = vec.fit_transform(dataset_d)

    return dataset_t, vec.feature_names_


def conform_data(dataset, targets, test_prop=0.30):
    """This function takes the dataset returned by 'transform_dataset'
    function and returns the trainer and tests arrays returned by
    sklearn.cross_validation module.
    """
    from sklearn.cross_validation import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, targets, test_size=test_prop, random_state=42)

    return X_train, X_test, y_train, y_test


def autoformat_element(element):
    """Tries to convert the passed parameter in float format.
    If the input parameter is of str type then only a sanitization of str is
    performed.
    """
    def str_sanitization(element):
        """Erases the '\n' at the end in some str objects"""
        return element.split('\n')[0]

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
