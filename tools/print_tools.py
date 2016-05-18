import numpy
import pylab

def print_some_data(x, y, count=25):
    """Prints the number count of data from x and y
    If labels = 'all' prints all data from x. If not
    only prints the data with the labels in labels.
    """
    for idx in range(0, count):
        try:
            print 'x[{0}]: {1}'.format(idx, x[count])
            print 'y[{0}]: {1}'.format(idx, y[count])
        except IndexError:
            print 'Not enough elements'
    """
    TODO: Print only labels in 'labels' parameter.
    """
    return


def print_from_dataset(dataset, count=5):
    """
    """
    for idx in xrange(count):
        print 'dataset[{0}] : {1}'.format(idx, dataset[count])
    print 'Total elements in row: {}'.format(len(dataset[0]))
    return


def print_data(dataset, unique=None):
    """This function receives an object or container and tries to print in
    a pretty format readable by the user, using PrettyPrinter library.
    """
    from pprint import PrettyPrinter
    printer = PrettyPrinter()
    if unique:
        printer.pprint(dataset[unique])
    else:
        printer.pprint(dataset)


def inspect_dataset(dataset_d):
    """Helper function used to print certain elements from the dataset.
    """
    count = len(dataset_d)
    labels = dataset_d[0].keys()
    for idx in xrange(count):
        for label in labels:
            if dataset_d[idx][label] == '':
                print dataset_d[idx]
                break
    return


def print_random_elements(arr, count=10):
    """Helper function which prints count random elements from arr."""
    import random
    arr_len = len(arr)
    if arr_len > 0:
        for idx in xrange(count):
            print arr[random.randint(0, arr_len)]
    # Else: Not enough elements to print
    return


def print_threshold_element(dataset_d, threshold, label):
    """Helper function used to print certain elements from the dataset.
    """
    count = len(dataset_d)
    for idx in xrange(count):
        if dataset_d[idx][label] > threshold:
            print dataset_d[idx]
    return


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


def plot_with_bars(labels, features, plot_count=10):
    """This function plots the data in features param in a bar chart. 
    """
    sorted_idx = numpy.argsort(features)
    bar_position = numpy.arange(sorted_idx.shape[0]) + .5
    pylab.barh(bar_position, features[sorted_idx], align='center')
    pylab.yticks(bar_position, labels[sorted_idx])
    pylab.xlabel('Features importances')
    pylab.show()
    return