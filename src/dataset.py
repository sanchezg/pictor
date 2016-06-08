import numpy as np
from sklearn.feature_extraction import DictVectorizer
from helper_functions import autoformat_element


def labels_sanitization(labels):
    labels_t = labels[:]
    for idx in xrange(len(labels)):
        if labels[idx] == None:
            labels_t[idx] = ''
    return labels_t


class Dataset(object):
    def __init__(self, csv_filename, f_discard, empty_val=0):
        self.labels = []
        self.dataset = []
        self.targets = []
        self.load_from_file(csv_filename, empty_val)
        if f_discard is not None:
            self.discard_features(f_discard)

    # Basic load methods
    def load_from_file(self, filename, empty_val, delimiter='|'):
        """Tries to load the dataset from a csv file.
        The function loads the dataset as a list of dicts and the
        labels as a list of str."""
        self.labels = []
        self.dataset = []
        first_line_loaded = False
        try:
            file_obj = open(filename, 'r')
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            return [], []

        lines = file_obj.readlines()
        file_obj.close()

        for line in lines:
            line_s = line.split(delimiter)
            data_line = map(autoformat_element, line_s,
                            [empty_val] * len(line_s))
            if first_line_loaded:
                inner_dict = dict(zip(self.labels, data_line))
                self.dataset.append(inner_dict)
            else:
                first_line_loaded = True
                self.labels = labels_sanitization(data_line)

    def discard_features(self, features_unwanted):
        """Removes from self.dataset those features which labels are in
        features_unwanted.
        self.dataset should be a list of dicts as the one returned from
        'load_dataset_from_csv'.
        features_unwanted should be a list of labels in str format."""
        for row in self.dataset:
            for feature in features_unwanted:
                try:
                    del row[feature]
                except KeyError:
                    pass

    # Basic feature engineering
    def impute_feature_value(self, feature, candidate, fn):
        """Replaces all values from the dataset were the feature is lower or
        higher than 'candidate' or is equal to None (missing value) with the
        candidate value. 'fn' determines the function replacement (lower_than,
        higher_than or is_None)."""
        count = 0
        for row in self.dataset:
            if fn(row[feature], candidate):
                row[feature] = candidate
                count += 1
        return count

    def remove_outliers(self, feature, threshold, fn):
        """Removes those samples from the dataset were the value of feature
        'feature' is higher or lower than the threshold."""
        for row in self.dataset:
            if fn(row[feature], threshold):
                del row

    def split_dataset(self, target_feature='interactions'):
        """Removes from the input dataset targets values (output), and appends
        them to other inner list."""
        self.targets = []
        for row in self.dataset:
            self.targets.append(row.pop(target_feature))

    def transform_dataset(self):
        """Uses sklearn DictVectorizer to transform the dataset and convert
        inner categorical features in a suitable representation.
        Warning: Once this method is called, self.dataset changes it's inner
        form."""
        print self.dataset[0]
        vec = DictVectorizer(sparse=False)
        self.dataset = vec.fit_transform(self.dataset)
        self.feature_names = vec.feature_names_

    def analyse_continuous_feature(self, feature):
        """Analyse information related to the numerical feature as amount of
        samples with valid values, mean, median and std deviation
        for each feature, maximum and minimum, and 1st and 3rd Q."""
        missing = 0
        valid_samples = []
        for sample in self.dataset:
            if sample[feature] is not None:
                # Only compute valid samples
                valid_samples.append(sample[feature])
            else:
                missing += 1

        if len(valid_samples) == 0:
            return None

        return (missing, np.mean(valid_samples), np.median(valid_samples),
                np.std(valid_samples), np.amax(valid_samples),
                np.amin(valid_samples), np.percentile(valid_samples, 25),
                np.percentile(valid_samples, 75))

    def analyse_categorical_feature(self, feature):
        """Analyse and returns information related to the categorical feature
        as amount of samples with valid values, 1st and 2nd modes, frequencies
        of 1st and 2nd modes, and cardinality."""
        valid_samples = 0
        count_values = {}
        for sample in self.dataset:
            # Lookup feature value for each sample
            if feature in sample.keys() and isinstance(sample[feature], str):
                category_val = sample[feature]
                valid_samples += 1
                if category_val in count_values.keys():
                    count_values[category_val] += 1
                else:
                    count_values[category_val] = 1
        if valid_samples == 0:
            return None

        # Transform into a list of tuples ordered by count
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

        return (valid_samples, len(count_values), mode1, fmode1, mode2,
            fmode2)

    def replace_rgb_features(self):
        """Removes the 'media_color_red|green|blue' features and puts a new
        feature called 'media_rgb'."""
        for idx in xrange(len(self.dataset)):
            sample = self.dataset[idx]
            self.dataset[idx]['media_rgb'] = None
            if (sample['media_color_blue'] is not None) and (
                sample['media_color_green'] is not None) and (
                sample['media_color_red'] is not None):
                new_val = (sample['media_color_blue'] + sample['media_color_green']
                                + sample['media_color_red'])/3.
                self.dataset[idx]['media_rgb'] = new_val
                del self.dataset[idx]['media_color_red']
                del self.dataset[idx]['media_color_green']
                del self.dataset[idx]['media_color_blue']

    def get_continuous_features(self):
        return ['caption_char_lenght', 'caption_hash_count',
            'caption_hash_ratio', 'caption_non_alpha_count',
            'caption_upper_count', 'caption_world_length', 'checkouts',
            'days', 'hashtags_count',
            'hashtags_ontology_frequency',
            'hashtags_ontology_reach', 'hashtags_reach_score',
            'how_many_faces', 'media_age_on_system',
            'media_bright_coefficient_variation', 'media_bright_variance',
            'media_rgb',
            'media_comments_count', 'media_hue_coefficient_variation',
            'media_hue_variance', 'media_likes_count', 'media_pixels',
            'media_saturation_coefficient_variation', 
            'media_saturation_variance'
            ]

    def get_categorical_features(self):
        return ['customer', 'hashtags_category', 'hashtags_count_segmented',
            'hashtags_encoding', 'hashtags_language',
            'location_longitude_category', 'location_country',
            'location_longitude_category', 'location_country',
            'media_filter', 'media_orientation', 'media_source', 'status',
            'vertical'
            ]

    def get_targets(self):
        """Returns a copy of local targets (interactions)."""
        return self.targets

    def get_dataset(self):
        """Returns a copy of local dataset."""
        return self.dataset

    def get_feature_names(self):
        """Returns a list with feature names."""
        return self.dataset[0].keys()

    def get_sample(self, idx):
        """Returns all features values for a specific sample."""
        return self.dataset[idx]

    def get_amount_samples(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)


if __name__ == '__main__':
    print "Please do not call this file directly."
