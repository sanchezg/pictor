#!/usr/bin/python

import sys
from time import time

sys.path.append('.')
from format_tools import load_dataset_from_csv, plot_data
from format_tools import discard_features, preformat_dataset
from format_tools import transform_dataset, split_dataset, conform_data
from format_tools import discard_outliers, count_equal_elements
from regression_tools import make_prediction
from print_tools import print_some_data, print_from_dataset, inspect_dataset
from print_tools import print_random_elements, print_threshold_element

corpus_dataset = []

if __name__ == '__main__':
    print 'Loading dataset from csv file...'
    t0 = time()
    corpus_dataset, corpus_labels = load_dataset_from_csv(
                                    '../consolidated_features.csv')
    print 'Time loading dataset: ', time() - t0

    features_unwanted = []
    features_unwanted.append('caption_char_lenght')
    # features_unwanted.append('caption_hash_count')
    features_unwanted.append('caption_hash_ratio') # Dependant variable
    features_unwanted.append('caption_language')
    features_unwanted.append('caption_non_alpha_count')
    features_unwanted.append('caption_upper_count')
    features_unwanted.append('caption_world_length')
    features_unwanted.append('customer')
    features_unwanted.append('customer_id')
    features_unwanted.append('days')
    features_unwanted.append('faces_detected')
    features_unwanted.append('has_nudity')
    features_unwanted.append('hashtags_category_count')
    features_unwanted.append('hashtags_count_segmented')
    features_unwanted.append('hashtags_count')
    features_unwanted.append('hashtags_encoding')
    features_unwanted.append('hashtags_most_frequent_jaccard')
    features_unwanted.append('hashtags_most_popular_jaccard')
    features_unwanted.append('hashtags_most_popular_similarity')
    features_unwanted.append('hashtags_ontology_reach')
    features_unwanted.append('hashtags_ratio_most_frequent')
    features_unwanted.append('hashtags_ratio_most_popular')
    features_unwanted.append('how_many_faces')
    features_unwanted.append('image_url')
    features_unwanted.append('location_longitude_category')
    # features_unwanted.append('location_country')
    features_unwanted.append('media_age_on_system')
    features_unwanted.append('media_bright_coefficient_variation')
    features_unwanted.append('media_bright_variance')
    features_unwanted.append('media_color_blue')
    features_unwanted.append('media_color_green')
    features_unwanted.append('media_color_red')
    features_unwanted.append('media_filter')
    features_unwanted.append('media_hue_coefficient_variation')
    features_unwanted.append('media_hue_variance')
    features_unwanted.append('media_id')
    features_unwanted.append('media_orientation')
    features_unwanted.append('media_pixels')
    features_unwanted.append('media_saturation_coefficient_variation')
    features_unwanted.append('media_saturation_variance')
    features_unwanted.append('Unnamed: 0') # Additional label 
    features_unwanted.append('') # Element order

    print 'Discarding undesired features...'
    t0 = time()
    discard_features(corpus_dataset, features_unwanted)
    print 'Time discarding features: ', time() - t0

    print 'Preformatting dataset...'
    t0 = time()
    preformat_dataset(corpus_dataset)
    print 'Time preformatting dataset: ', time() - t0

    # inspect_dataset(corpus_dataset)
    # print_threshold_element(corpus_dataset, 30000., 'interactions')
    # count_equal_elements(corpus_dataset, 'caption_hash_count', 'hashtags_count')
    # count_equal_elements(corpus_dataset, 'hashtags_most_frequent_jaccard',
    #     'hashtags_most_frequent_similarity')
    # count_equal_elements(corpus_dataset, 'hashtags_most_popular_jaccard',
    #     'hashtags_most_popular_similarity')
    # count_equal_elements(corpus_dataset, 'hashtags_ontology_reach',
    #     'hashtags_reach_score')

    # print 'Discarding outliers...'
    # t0 = time()
    # discard_outliers(corpus_dataset, 30000., 'interactions')
    # print 'Time discarding outliers: ', time() - t0

    print 'Splitting targets...'
    t0 = time()
    targets = split_dataset(corpus_dataset)
    print 'Time splitting dataset: ', time() - t0

    print 'Transforming dataset...'
    t0 = time()
    dataset = transform_dataset(corpus_dataset)
    print 'Time transforming dataset: ', time() - t0

    print 'Conforming training and testing arrays...'
    t0 = time()
    X_train, X_test, y_train, y_test = conform_data(dataset, targets)
    print 'Time conforming data: ', time() - t0

    # print_some_data(X_train, y_train, count=1)
    # plot_data('media_likes_count', 'interactions', data[a0:a1])

    print 'Beggining with predictions...'
    make_prediction(X_train, y_train, X_test, y_test, show_score=True,
        slice_samples=20)

    # print_random_elements(y_train, count=500)
