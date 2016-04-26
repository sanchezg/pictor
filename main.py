#!/usr/bin/python

import sys
from time import time

sys.path.append('.')
from format_tools import load_data_from_csv, print_data, format_data, plot_data
from format_tools import convert_to_float, format_input_data, classify_elements
from format_tools import list_from_features, conform_data
from print_tools import print_some_data

corpus_dataset = []

def explore_features(dataset, features):
    return

if __name__ == '__main__':
    t0 = time()
    corpus_dataset, corpus_labels = load_data_from_csv('../consolidated_features.csv')
    print 'Time loading dataset: ', time()-t0

    # exc_features = list_from_features(corpus_dataset, ['interactions', 'media_likes_count'])
    # features, target = format_input_data(corpus_dataset, 'media_likes_count', 'interactions')
    # labels_used = ['media_id', 'media_likes_count',
    #     'checkouts', 'days', 'media_comments_count', 'vertical',
    #     'media_source', 'hashtags_category', 'hashtags_most_frequent_similarity',
    #     'hashtags_most_popular_jaccard', 'hashtags_most_popular_similarity',
    #     'caption_hash_count'
    #     ]
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

    print_some_data(X_train, y_train)
    # print 'Features len: ', len(features)
    # data = []
    # for idx in range(0, len(features)):
    #     # discard outliers
    #     if features[idx] < 68000:
    #         data.append([features[idx], target[idx]])
    # print 'Data len: ', len(data)

    # a0 = 0
    # a1 = len(data)/100
    # a2 = 2*a1
    # a3 = 3*a1
    # a4 = 4*a1
    # a5 = 5*a1
    # a6 = 6*a1
    # a7 = 7*a1
    # a8 = 8*a1
    # a9 = 9*a1
    # a10 = 10*a1

    # plot_data('media_likes_count', 'interactions', data[a0:a1])
    # data_classified = classify_elements(corpus_dataset, '')
    # print 'Length classification: ', len()
