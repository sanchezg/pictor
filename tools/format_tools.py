#!/usr/bin/python

import numpy

"""This file provides functions and tools for formatting data into different
options."""


def load_dataset_from_csv(filename, delimiter='|', first_line=True,
                          output_format='dict'):
    """This function tries to load a list of dictionaries from a csv file.
    Parameter 'first_line' indicates if the first line in the csv file
    contains the labels for the dataset. These labels are used as keys
    in the dictionary.
    The function returns the dataset (as a list of dict) and the labels
    (as a list of str).
    """
    from time import time
    print 'Loading dataset from csv file...'
    t0 = time()

    labels = []
    dataset = []
    first_line_loaded = False
    try:
        file_obj = open(filename, 'r')
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        return

    # CSV data will be loaded as:
    # 'key': {
    #        'ikey1': value,
    #         'ikey2': value,
    #        ...
    #        'ikeyn': value
    #        }

    for line in file_obj.readlines():
        data_line = line.split(delimiter)
        if first_line_loaded:
            inner_dict = dict(zip(labels, data_line))
            dataset.append(inner_dict)
        else:
            labels = data_line
            first_line_loaded = True
    print "Time loading dataset: {}".format(time() - t0)
    return dataset, labels


def load_features_from_file(filename='features_unwanted.csv'):
    """This function loads a list with features names from the passed
    argument.
    """
    features_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        features_list = [str(feat).split('\n')[0] for feat in lines
                            if not feat.startswith('#')]
    if 'vertical' in features_list:
        idx = features_list.index('vertical')
        features_list[idx] = 'vertical\n'
    # print features_list
    return features_list


def discard_features(dataset_d, features_unwanted):
    """This function removes from dataset_d all features which labels are in
    features_unwanted.
    dataset_d should be a list of dicts as the one returned from
    'load_dataset_from_csv'.
    features_unwanted should be a list of labels in str format.
    """
    from time import time

    print 'Discarding undesired features...'
    t0 = time()

    for row in dataset_d:
        for feature in features_unwanted:
            try:
                del row[feature]
            except KeyError:
                pass
    print "Time discarding features: {}".format(time() - t0)
    return


def preformat_dataset(dataset_d):
    """This function takes the dataset loaded with 'load_dataset_from_csv'
    function and converts all numerical values to float type.
    """
    from time import time

    print 'Preformatting dataset...'
    t0 = time()

    count = len(dataset_d)
    labels = dataset_d[0].keys()
    for idx in xrange(count):
        for label in labels:
            dataset_d[idx][label] = autoformat_element(dataset_d[idx][label])
    print "Time preformatting dataset: {}".format(time() - t0)
    return


def scale_dataset(corpus_dataset):
    """
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
    from time import time

    print 'Splitting targets...'
    t0 = time()

    output_values = []
    for row in dataset_d:
        output_values.append(row.pop(target_feature))

    print "Time splitting dataset: {}".format(time() - t0)
    return output_values


def transform_dataset(dataset_d):
    """Uses sklearn DictVectorizer to transform the dataset and convert inner
    categorical features in a suitable representation.
    """
    from time import time
    from sklearn.feature_extraction import DictVectorizer
    print 'Transforming dataset...'
    t0 = time()
    vec = DictVectorizer(sparse=False)
    dataset_t = vec.fit_transform(dataset_d)
    print "Time transforming dataset: {}".format(time() - t0)
    return dataset_t


def conform_data(dataset, targets, test_prop=0.30):
    """This function takes the dataset returned by 'transform_dataset'
    function and returns the trainer and tests arrays returned by
    sklearn.cross_validation module.
    """
    from time import time
    from sklearn.cross_validation import train_test_split

    print 'Conforming training and testing arrays...'
    t0 = time()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, targets, test_size=test_prop, random_state=42)
    print "Time conforming data: {}".format(time() - t0)
    return X_train, X_test, y_train, y_test


def format_data(dataset_d, target_label, exc_features=[]):
    """This function receives a dictionary formatted with load_data_from_csv
    function and returns a pair of features and targets list of values.
    Parameter 'dataset_d' is the dictionary mention before.
    Parameter 'target_label' is the label desired as output (target).
    Parameter 'exc_features' is a list of features to exclude in the
    returned lists.
    """
    target = []
    elements = len(dataset_d)
    features = [[]]
    # obtain keys of dictionary
    keys = dataset_d[0].keys()

    # be sure that target label will not be in features labels
    exc_features_t = exc_features
    exc_features_t.append(target_label)

    for element_idx in xrange(elements):
        # element is a dict with all features as keys and result of feature as
        # data.
        values = [float(dataset_d[element_idx][key])
                  for key in keys if key not in exc_features_t]
        features.append(values[0])
        target.append(float(dataset_d[element_idx][target_label]))
    return features, target


def plot_data(x_label, y_label, data, color='b'):
    """This function receives a list of points and labels and plots them using
    matplot library.
    Data is an array of arrays in the form:
        data[[point_x][point_y]].
    """
    import matplotlib.pyplot

    for point in data:
        point_x = point[0]
        point_y = point[1]
        matplotlib.pyplot.scatter(point_x, point_y)

    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.show()
    return


def discard_outliers(dataset, threshold, label='interactions'):
    """Removes from the input dataset those rows with element bigger than
    threshold.
    """
    dataset_t = dataset
    idx = 0
    for row in dataset:
        if row[label] > threshold:
            del dataset_t[idx]
        idx += 1
    dataset = dataset_t[:]
    return


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


def count_equal_elements(dataset, label1, label2):
    """Helper function: counts features with equal values."""
    count_equals = 0
    count_l1_zero = 0
    count_l2_zero = 0
    count_distinct = 0
    for row in dataset:
        if row[label1] == row[label2]:
            count_equals += 1
        elif row[label1] == 0 and row[label2] != 0:
            count_l1_zero += 1
        elif row[label1] != 0 and row[label2] == 0:
            count_l2_zero += 1
        else:
            count_distinct += 1

    print 'Total {0} elements equals to {1} elements: {2}'.format(
        label1, label2, count_equals)
    print 'Total {0} elements totally distincts to {1} elements: {2}'.format(
        label1, label2, count_distinct)
    print 'Total {0} elements equals to zero: {1}'.format(
        label1, count_l1_zero)
    print 'Total {0} elements equals to zero: {1}'.format(
        label2, count_l2_zero)
    return
