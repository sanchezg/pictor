import argparse


def get_args_parsed():
    """Reads the input arguments passed to the program."""
    def process_arguments():
        """Creates an ArgumentParser object and process the inputs arguments.
        """
        parser = argparse.ArgumentParser(
            description='Pictor. More info: https://github.com/sanchezg/pictor')
        parser.add_argument('--csv', type=str,
                            help='Path to input csv with dataset')
        parser.add_argument('--feat', type=str,
                            help='Path to file with features to discard')
        args = parser.parse_args()
        return args
    args = process_arguments()
    # Default input values
    csv_filename = 'consolidated_features.csv'
    feat_filename = None
    if args.csv is not None:
        csv_filename = args.csv
    if args.feat is not None:
        feat_filename = args.feat
    return csv_filename, feat_filename


def file_to_list(filename):
    """This function loads a list with features names from the file passed as
    argument."""
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
        features_list[idx] = 'vertical\\n'
    return features_list


def autoformat_element(element, empty_val=0):
    """Tries to convert the passed parameter in float format.
    If the input parameter is of str type then only a sanitization of str is
    performed."""
    def str_sanitization(element):
        """Erases the '\n' at the end in some str objects"""
        return element.split('\n')[0]

    try:
        # Some elements can be empty
        if element == '':
            element_formatted = empty_val
        else:
            element_formatted = float(element)
    except ValueError:
        # The element is of str type
        element_formatted = str_sanitization(element)
    return element_formatted
