import numpy


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
    """Helper function used to print those elements from the dataset which
    value is null.
    """
    count = len(dataset_d)
    labels = dataset_d[0].keys()
    for idx in xrange(count):
        for label in labels:
            if dataset_d[idx][label] == '':
                print dataset_d[idx]
                break
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

    point_x = 0
    for point_y in sorted(data):
        matplotlib.pyplot.scatter(point_x, point_y)
        point_x += 1

    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.show()
    return


def plot_histogram(x_label, y_label, data, normed=False, step=100):
    """Plots an histogram with the array data passed by parameter.
    """
    import matplotlib.pyplot

    bins = [a for a in xrange(0, int(max(data))+step, step)]
    n, bins, p = matplotlib.pyplot.hist(data, bins=bins, normed=normed)

    print "Histogram results..."
    print "bin, n: {}".format(zip(bins, n))

    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.show()
    return


def plot_with_bars(labels, importances, plot_count=10):
    """This function plots the data in importances param in a bar chart.
    """
    import matplotlib.pyplot as plt

    pair_sorted = sorted(zip(importances, labels))

    end_slice = 0
    if isinstance(plot_count, int) and plot_count > 0:
        end_slice = -plot_count

    imp_slice = []
    lab_slice = []
    max_importance = max(importances)
    for value, label in pair_sorted[end_slice:]:
        lab_slice.append(label)
        imp_slice.append(value/max_importance)

    # plt.rcdefaults()
    ypos = numpy.arange(len(lab_slice)) + .5

    plt.barh(ypos, imp_slice, align='center', alpha=0.5)
    plt.yticks(ypos, lab_slice)
    # plt.xticks(ypos, lab_slice)
    plt.xlabel('Importance')
    plt.title('Features importance')

    plt.show()
    return
