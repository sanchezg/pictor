import numpy as np
from sklearn.feature_extraction import DictVectorizer
from helper_functions import autoformat_element


def labels_sanitization(labels):
    labels_t = labels[:]
    for idx in xrange(len(labels)):
        if labels[idx] == None:
            labels_t[idx] = ''
    return labels_t


class DatasetExplorer(object):
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

    def remove_feature_none(self):
        """Removes all the features without name."""
        for row in self.dataset:
            try:
                del row[None]
            except KeyError:
                pass

    # Basic feature engineering
    def impute_feature_value(self, feature, candidate, fn):
        """Replaces all values from the dataset were the feature is lower or
        higher than 'candidate' or is equal to None (missing value) with the
        candidate value. 'fn' determines the function replacement (lower_than,
        higher_than or is_None)."""
        count = 0
        for sample in self.dataset:
            if fn(sample[feature], candidate):
                sample[feature] = candidate
                count += 1
        return count

    def remove_samples(self, feature, candidate, fn):
        """Removes those samples from the dataset were the value of feature
        'feature' is higher or lower than the threshold."""
        count = 0
        for row in self.dataset:
            if fn(row[feature], candidate):
                del row[feature]
                count += 1
        return count

    def split(self, target_feature='interactions', validation=0.3):
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

    def lookup_modes(self, feature):
        """Returns a list of tuples with modes names and modes count for
        the categorical feature passed by argument."""
        valid_samples = 0
        count_values = {}
        for sample in self.dataset:
            # Lookup feature value for each sample
            if feature in sample.keys() and isinstance(sample[feature],
                                                       str):
                category_val = sample[feature]
                valid_samples += 1
                if category_val in count_values.keys():
                    count_values[category_val] += 1
                else:
                    count_values[category_val] = 1
        if valid_samples == 0:
            return None
        # Transform into a list of tuples, and order by count
        count_values = count_values.items()
        count_values.sort(key=lambda e: e[1])
        return count_values, valid_samples

    def analyse_categorical_feature(self, feature):
        """Analyse and returns information related to the categorical feature
        as amount of samples with valid values, 1st and 2nd modes, frequencies
        of 1st and 2nd modes, and cardinality."""
        cv = self.lookup_modes(feature)
        if cv is None:
            return None
        count_values, valid_samples = cv
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

    def replace_rgb_feature(self):
        """Removes the 'media_color_red|green|blue' features and puts a new
        feature called 'media_rgb'."""
        for sample in self.dataset:
            sample['media_rgb'] = None
            new_val = 0.
            div = 0
            try:
                if sample['media_color_blue'] is not None:
                    new_val += sample['media_color_blue']
                    div += 1
                if sample['media_color_green'] is not None:
                    new_val += sample['media_color_green']
                    div += 1
                if sample['media_color_red'] is not None:
                    new_val += sample['media_color_red']
                    div += 1
                if div > 0:
                    new_val = new_val/div
                else:
                    new_val = None
                sample['media_rgb'] = new_val
                del sample['media_color_red']
                del sample['media_color_green']
                del sample['media_color_blue']
            except KeyError:
                pass

    def feature_is_continuous(self, feature):
        """Returns True if the feature passed by argument is continuous.
        Returns False otherwise."""
        fcont = True
        for row in self.dataset:
            if isinstance(row[feature], str):
                fcont = False
        return fcont

    def format_categorical_feature(self, feature):
        """Analyse and formats 'feature' looking for modes with value less or
        equal to 2."""
        cv = self.lookup_modes(feature)
        if cv is None:
            return None
        count_values, valid_samples = cv
        modes_list = [mode for mode, count in count_values if count<=2]
        if len(modes_list) == 0:
            return None
        count = 0
        new_mode = '-'.join(modes_list) # Define a new mode
        for mode in modes_list:
            for sample in self.dataset:
                if sample[feature] == mode:
                    sample[feature] = new_mode[:min(len(new_mode), 40)]
                    count += 1
        return count

    def format_customer_values(self):
        """Formats the values for customer feature, keeping only those
        different to the ones indicated here."""
        customer_values = ["aeropostale", "ann taylor", "clinique", "dwr",
            "keds", "macy's fashion", "reebok", "tomford", "tweenbrands",
            "warrior hockey", "[inactive] walmartt", "maybelline - unused",
            "puma spain (receiver)", "maybelline2",
            "puma austria (receiver) ",
            "puma luxembourg (receiver) -zumba fitness", "mini cooper",
            "[walmart] department - electronics ", "macy's beauty",
            "puma belgium (receiver)", "est\xc3\xa9e lauder", "ralph lauren",
            "crate and barrel", "live the look", "walmart bhm migration",
            "target", "texas tourism", "puma italy (receiver)",
            "americanairlines", "#stubhubtd", "20jeans", "@piperlime",
            "[global] redken", "[regional] redken - us", "free people",
            "[walmart] department - baby", "[walmart] department - beauty",
            "[walmart] master", "[walmart] seasonal - crafts central",
            "adidas", "afloral", "alex and ani", "american eagle",
            "americaneaglemigration", "aritzia - ca - en", "aritzia - ca - fr",
            "aritzia - us - en", "aritziamaster@olapic.com", "baltimore ravens ",
            "bare escentuals - inactive", "baukjen official ", "bcbg",
            "bedbathandbeyond", "bet", "black lapel", "bob mills furniture",
            "boticca", "bowling green state university", "brooks running eu",
            "bucketfeet", "bumble & bumble", "carhartt", "cath kidston",
            "charm and chain", "christian louboutin", "coach",
            "coach mothers day", "coastal", "coffeebean", "columbia",
            "crocs us account", "dagne dover", "dailylook", "dannijo",
            "darby smart", "deb prom", "deb shops", "desigual", "dollskill",
            "drmartens", "dvf", "eb games", "electric frenchie", "em cosmetics",
            "equipment", "frank & eileen", "general pants", "giggle", "guess",
            "gypsy warrior", "hard rock cafe", "harman kardon", "ily couture",
            "isabella oliver migration", "isabella oliver official", "ivivva",
            "j brand", "j. crew", "jack rogers", "jbl", "jetblue",
            "jewelers wife", "jewelmint", "), kmart fashion", "lancome",
            "lilly pulitzer", "loreal", "loren hope", "lulu frost", "lululemonv2",
            "macy's idea lab", "macy's shoes", "madewell", "madewell denim",
            "make up for ever", "make up for ever france",
            "make up for ever international", "maybelline ",
            "maybelline ca - brows", "maybelline canada", "maybelline ru",
            "maybelline uk", "men's wearhouse", "miansai", "monki",
            "motives cosmetics ", "mountain hardwear", "moving comfort", "murad",
            "nasty gal", "new steve madden", "newton running",
            "ninashoes@olapic.com", "nine west", "oclarisonic", "oiselle",
            "old steve madden", "onward reserve ", "pacsun", "peak performance",
            "peek kids", "pink blush maternity", "poppin", "puma (sender)",
            "puma france (reciever)", "puma germany (receiver)",
            "puma switzerland (receiver) ", "puma united kingdom (receiver)",
            "pura vida", "quiksilver", "reeds@olapic.com", "reef", "rockport",
            "ruum american kid's wear", "sally beauty", "saucony", "scottevest",
            "shoemint", "shop planet blue", "showpo", "sidewalk district",
            "silver jeans", "simple", "simple skin care", "soludos", "sony",
            "sourpuss clothing", "southern tide", "spanx", "sperry europe",
            "splits59@olapic.com", "step2", "stride rite", "stubhub playon!",
            "stylemint", "tacori", "tarte", "tea collection", "telva", "teva",
            "the mountain", "the north face", "threadless", "tilly's",
            "tilted sole", "toms shoes", "too faced", "topo", "topshop uk",
            "total hockey", "tribesports", "tuckernuck", "tula", "tumi",
            "uncle jeans", "urban decay", "usa track & field", "vans",
            "veuve clicquot ", "vince camuto", "vineyard vines", "vizio",
            "waxing poetic", "west elm", "westward leaning", "wildfox couture",
            "wool & the gang", "world kitchen", "zulily"]
        new_value = "others"
        count = 0
        for sample in self.dataset:
            if sample['customer'] in customer_values:
                sample['customer'] = new_value
                count += 1
        return count

    def format_many_modes(self, feature, mcount=3):
        """Keeps only 4 modes in the feature, the first 3 and an 'other'
        value."""
        if feature not in self.dataset[0].keys():
            return -2
        cv = self.lookup_modes(feature)
        if cv is None:
            return -1
        count_values, valid_samples = cv
        if len(count_values) < 5:
            return 0
        # Keep only with best three modes
        modes = [mode for mode, count in count_values[-3:]]
        new_value = "other"
        for sample in self.dataset:
            if sample[feature] not in modes:
                sample[feature] = new_value

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
