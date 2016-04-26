def print_some_data(x, y, labels='all', count=25):
    """Prints the number count of data from x and y
    If labels = 'all' prints all data from x. If not
    only prints the data with the labels in labels.
    """
    if labels == 'all':
        for idx in range(0, count):
            try:
                print 'x[{0}]: {1}'.format(idx, x[idx])
                print 'y[{0}]: {1}'.format(idx, y[idx])
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