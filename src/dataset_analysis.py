import numpy as np
from time import time
import dataset
from helper_functions import *
from plotter_functions import plot_histogram, plot_scatter


def plot_histogram_features(data, feature):
    """Plots a histogram for each feature continuous in 'feat_continuous'
    contained in dataset 'data'."""
    print "Plotting histogram for: '{}'.".format(feature)
    if len(data) > 0 and feature in ['hashtags_most_frequent_jaccard',
                                     'hashtags_most_frequent_similarity',
                                     'hashtags_most_popular_jaccard']:
        # Features with high values should be normalized
        normed = True
    else:
        normed = False

    bins_count = plot_histogram(feature, 'number', data, normed=normed,
                                save=True, aname='_post')
    print "Histogram results..."
    print "bin, n: {}".format(bins_count)


def get_statistics(dataset, feature):
    """Returns the statistical information for the feature 'feature' in the
    dataset 'dataset'."""
    feat_values = []
    for idx in xrange(dataset.get_amount_samples()):
        sample = dataset.get_sample(idx)
        if feature in sample.keys() and isinstance(sample[feature], float):
            feat_values.append(sample[feature])

    if len(feat_values) == 0:
        return None

    return (feat_values, np.mean(feat_values), np.median(feat_values),
        np.std(feat_values), np.amax(feat_values), np.amin(feat_values),
        np.percentile(feat_values, 25), np.percentile(feat_values, 75))


def analyse_continuous(ds, features):
    """Prints to the console information related to each numerical feature;
    as amount of samples with valid values, mean, median and std deviation
    for each feature, maximum and minimum, and 1st and 3rd Q."""
    print "Total samples: {}".format(ds.get_amount_samples())
    print "Features: {}".format(ds.get_feature_names())
    for feat in features:
        print "======= Feature '{}' analysis:".format(feat)
        stat_results = get_statistics(ds, feat)
        if stat_results is not None:
            print "Samples with values: {}".format(len(stat_results[0]))
            print "Mean: {}".format(stat_results[1])
            print "Median: {}".format(stat_results[2])
            print "Std Dev: {}".format(stat_results[3])
            print "Max: {}".format(stat_results[4])
            print "Min: {}".format(stat_results[5])
            print "1st percentile: {}".format(stat_results[6])
            print "3rd percentile: {}".format(stat_results[7])
            print "Cardinality: {}".format(len(set(stat_results[0])))
            plot_histogram_features(stat_results[0], feat)


def get_classifications(dataset, feature):
    feat_values = []
    count_values = {}
    for idx in xrange(dataset.get_amount_samples()):
        # Lookup feature value for each sample
        sample = dataset.get_sample(idx)
        if feature in sample.keys() and isinstance(sample[feature], str):
            category_val = sample[feature]
            feat_values.append(category_val)
            if category_val in count_values.keys():
                count_values[category_val] += 1
            else:
                count_values[category_val] = 1
    if len(feat_values) == 0:
        return None

    # Transform into a list of tuples
    count_values = count_values.items()
    count_values.sort(key=lambda e: e[1])

    # First and second modes
    if len(count_values) > 2:
        mode1, fmode1 = count_values[-1]
        mode2, fmode2 = count_values[-2]
    elif len(count_values) == 2:
        mode1, fmode1 = count_values[1]
        mode2, fmode2 = count_values[0]
    else:
        # Only one value:
        mode1, fmode1 = count_values[0]
        mode2, fmode2 = 0, 0

    return (len(feat_values), len(count_values), mode1, fmode1, mode2,
        fmode2, count_values)


def analyse_categorical(ds):
    """Prints to the console information related to each categorical feature;
    ."""
    print "Total samples: {}".format(ds.get_amount_samples())
    print "Features: {}".format(ds.get_feature_names())
    for feat in ds.get_feature_names():
        print "======= Feature: '{}' analysis:".format(feat)
        classification = get_classifications(ds, feat)
        if classification is not None:
            print "Samples with values: {}".format(classification[0])
            print "Cardinality: {}".format(classification[1])
            print "1st mode: {}".format(classification[2])
            print "1st mode freq: {}".format(classification[3])
            print "2nd mode: {}".format(classification[4])
            print "2nd mode freq: {}".format(classification[5])
            plot_bar_chart(classification[6], feat)


def analyse_continuous_x_features(ds, features_to_analyse):
    """Makes two lists for the features in features_to_analyse with samples
    from ds and calls the scatter plot function with both data."""
    xfeature = []
    yfeature = []
    for idx in xrange(ds.get_amount_samples()):
        sample = ds.get_sample(idx)
        if (sample[features_to_analyse[0]] is not None) and (
            sample[features_to_analyse[1]] is not None):
            xfeature.append(sample[features_to_analyse[0]])
            yfeature.append(sample[features_to_analyse[1]])
    plot_scatter(features_to_analyse[0], features_to_analyse[1],
                 xfeature, yfeature, save=True)


def calc_pearson_coeff(ds, xfeat, yfeat):
    """."""
    xfeature = []
    yfeature = []

    for idx in xrange(ds.get_amount_samples()):
        sample = ds.get_sample(idx)
        if (sample[xfeat] is not None) and (sample[yfeat] is not None):
            xfeature.append(sample[xfeat])
            yfeature.append(sample[yfeat])
    return np.corrcoef(xfeature, yfeature)


if __name__ == '__main__':
    # What to do if this module is executed directly
    csv_filename, feat_filename = get_args_parsed()
    print "Loading dataset..."
    t0 = time()
    ds = dataset.Dataset(csv_filename, file_to_list(feat_filename),
                         empty_val=None)
    print "Done. Time loading dataset: {:.2f}s".format(time() - t0)

    # analyse_continuous(ds)
    # analyse_categorical(ds)
    # Analyse and impute values for different features
    # features_to_analyse = ['caption_char_lenght', 'caption_hash_count',
    #     'caption_non_alpha_count', 'caption_upper_count', 'caption_world_length',
    #     'days', 'hashtags_count', 'hashtags_most_frequent_jaccard',
    #     'hashtags_most_frequent_similarity', 'hashtags_most_popular_jaccard',
    #     'hashtags_ontology_reach', 'media_bright_variance', 'media_comments_count',
    #     'media_likes_count', 'media_saturation_variance']
    # for feature in features_to_analyse:
    #     f_analysis = ds.analyse_continuous_feature(feature)
    #     if f_analysis is not None:
    #         missing = f_analysis[0]
    #         mean = f_analysis[1]
    #         median = f_analysis[2]
    #         std_dev = f_analysis[3]
    #         f_max = f_analysis[4]
    #         f_min = f_analysis[5]
    #         f_1q = f_analysis[6]
    #         f_3q = f_analysis[7]

    #         higher_thre = f_3q + (1.5 * (f_3q - f_1q))
    #         # Impute higher outliers
    #         c = ds.impute_feature_value(feature, higher_thre,
    #                                     lambda val, thre: val > thre)
    #         print "Imputed: {0} values for {1}".format(c, feature)

    # print "\n== Posterior imputing analysis"
    # analyse_continuous(ds, features_to_analyse)

    # features_to_analyse = ['checkouts', 'hashtags_most_popular_similarity']
    # print "Analysing {0} vs {1}".format(features_to_analyse[0],
    #     features_to_analyse[1])
    # analyse_continuous_x_features(ds, features_to_analyse)

    features_to_analyse = ['days', 'hashtags_count',
        'hashtags_ontology_frequency',
        'hashtags_ontology_reach', 'hashtags_ratio_most_frequent',
        'hashtags_ratio_most_popular',
        'hashtags_reach_score', 'how_many_faces', 'interactions',
        'media_age_on_system', 'media_bright_coefficient_variation',
        'media_bright_variance', 'media_comments_count',
        'media_hue_coefficient_variation', 'media_hue_variance',
        'media_likes_count', 'media_pixels',
        'media_saturation_coefficient_variation', 'media_saturation_variance',
        ]
    feature_name = 'media_color_red'

    for feat in features_to_analyse:
        cpearson = calc_pearson_coeff(ds, feature_name, feat)
        print "Pearson correlation coeff for {0} vs {1}:\n{2}".format(
                    feature_name, feat, cpearson)
