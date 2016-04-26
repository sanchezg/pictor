#!/usr/bin/python

import numpy

"""This file provides functions and tools for formatting data into different
options."""


def load_data_from_csv(filename, delimiter='|', first_line=True,
                       output_format='dict'):
    """This function tries to load a list of dictionaries from a csv file.
    Parameter 'first_line' indicates if the first line in the csv file
    contains the labels for the dataset. These labels are used as keys
    in the dictionary.
    The function returns the dataset (as a list of dict) and the labels
    (as a list of str).
    """
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
    return dataset, labels


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


def format_input_data(dataset, feature_label, target_label):
    """This function takes a dictionary returned by 'load_data_from_csv' and
    returns a list of pairs corresponding to the keys 'feature_label' and
    'target_label'.
    """
    targets = []
    features = []

    for element_idx in range(0, len(dataset)):
        feature_val = dataset[element_idx][feature_label]
        target_val = dataset[element_idx][target_label]
        try:
            features.append(float(feature_val))
            targets.append(float(target_val))
        except ValueError, e:
            pass
    return features, targets


def conform_data(dataset, labels, target_feat, test_prop=0.30):
    """This function takes the dataset returned by 'load_data_from_csv'
    function and returns the trainer and tests arrays returned by
    sklearn.cross_validation module.
    """
    from sklearn.cross_validation import train_test_split
    X = []
    y = []
    keys = dataset[0].keys()
    for element_idx in range(0, len(dataset)):
        X.append([
            dataset[element_idx][label] for label in keys
            if label in labels and label != target_feat
            ])
        y.append(dataset[element_idx][target_feat])
    return train_test_split(
        X, y, test_size=test_prop, random_state=42)


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


def classify_elements(dataset, interest_key, classifications_number):
    """This function classify a set of continuous features included in a
    dataset into a list of 'classifications_number' of classes.
    """
    keys = dataset.keys()
    if interest_key not in keys:
        print 'The feature is not in the dataset'
        return
    count_elements = len(dataset)
    interest_values = [
        dataset[idx][interest_key] for idx in xrange(count_elements)
        ]
    range_class = (max(interest_values) - min())/classifications_number
    list_class = []
    for idx in xrange(classifications_number):
        list_class.append(interest_values[
            idx*classifications_number:(idx+1)*classifications_number
            ])
    return list_class


def list_from_features(dataset, labels):
    """As dataset is a list of dicts, then we need to obtain keys from the
    first element.
    """
    return [key for key in dataset[0].keys() if key not in labels]


def discard_outliers(data_list, threshold):
    """Returns a list without the elements that are bigger than the threshold.
    """
    count = len(data_list)
    formatted = [
        data_list[idx] for idx in xrange(count) if data_list[idx] < threshold
        ]
    return formatted


def autoformat_element(element):
    """Tries to convert and return the passed parameter in float or int.
    """
    try:
        if '.' in element:
            element_formatted = float(element)
        else:
            element_formatted = int(element)
    except ValueError:
        element_formatted = element
    return element_formatted


def autoformat_list(data_list):
    """Tries to convert each element of the list passed as parameter in
    int or float. Otherwise it remains as is. The resulting list is returned.
    """
    list_formatted = [autoformat_element(element) for element in data_list]
    return list_formatted


def autoformat_np(X):
    """This function takes a numpy array and converts all numbers from str
    format in int or float. The resulting array is returned as numpy array.
    """
    X_conv = [autoformat_list(arr) for arr in X]
    return numpy.array(X_conv)
