#!/usr/bin/python

import sys
from time import time

sys.path.append('.')
from format_tools import load_data_from_csv, plot_data
from format_tools import conform_data, autoformat_np, autoformat_list
from print_tools import print_some_data

corpus_dataset = []

def explore_features(dataset, features):
    return

if __name__ == '__main__':
    t0 = time()
    corpus_dataset, corpus_labels = load_data_from_csv('../consolidated_features.csv')
    print 'Time loading dataset: ', time()-t0

    labels_used = corpus_labels
    labels_used.remove('media_source')
    labels_used.remove('hashtags_category')
    labels_used.remove('hashtags_ratio_most_popular')
    labels_used.remove('hashtags_encoding')
    labels_used.remove('faces_detected')
    labels_used.remove('media_filter')
    labels_used.remove('location_country')
    labels_used.remove('hashtags_ratio_most_frequent')
    labels_used.remove('hashtags_count_segmented')
    labels_used.remove('location_longitude_category')
    labels_used.remove('has_nudity')
    labels_used.remove('image_url')
    labels_used.remove('status')
    labels_used.remove('caption_language')
    labels_used.remove('media_orientation')
    labels_used.remove('hashtags_category_count')
    t0 = time()
    X_train, X_test, y_train, y_test = conform_data(corpus_dataset, labels_used, 'interactions')
    print 'Time conforming data: ', time() - t0

    # Autoconvert numerical values in int or float type
    X_train = autoformat_np(X_train)
    X_test = autoformat_np(X_test)
    y_train = autoformat_list(y_train)
    y_test = autoformat_list(y_test)

    print_some_data(X_train, y_train)

    # plot_data('media_likes_count', 'interactions', data[a0:a1])
