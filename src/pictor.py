import sys
from time import time
from sklearn.cross_validation import train_test_split
from predictor import Predictor
from dataset import DatasetExplorer
from helper_functions import get_args_parsed, file_to_list


def fill_empty_values(dataset):
    """Analyses continuous and categorical features and fills empty values for
    each feature with their correspondant mean (continuous) or mode
    (categorical)."""
    for f in dataset.get_feature_names():
        if dataset.feature_is_continuous(f):
            f_analysis = dataset.analyse_continuous_feature(f)
            if f_analysis is not None:
                mean = f_analysis[1]
                # Impute missing values with mean
                c = dataset.impute_feature_value(f, mean,
                                            lambda val, t: val==None)
                if c>0:
                    print "Imputed {0} values for feature {1}".format(c, f)
        else:
            # Analyse categorical features
            f_analysis = dataset.analyse_categorical_feature(f)
            if f_analysis is not None:
                mode1 = f_analysis[2]
                # Impute missing values with mean
                c = dataset.impute_feature_value(f, mode1,
                                            lambda val, t: val==None)
                if c>0:
                    print "Imputed {0} values for feature {1}".format(c, f)


# Execute if this module is called directly
if __name__ == '__main__':
    print "Welcome to pictor: a predictor for pictures interactions.\n"
    # Get program arguments
    csv_filename, features_filename = get_args_parsed()

    # Get features to discard from extern file
    if features_filename is not None:
        features_to_discard = file_to_list(features_filename)
    else:
        features_to_discard = None
    # Create a DatasetExplorer object and load with dataset
    print "Loading dataset from file..."
    t0 = time()
    dataset = DatasetExplorer(csv_filename, features_to_discard,
                              empty_val=None)
    print "Done. Time loading dataset: {:.2f}s".format(time()-t0)

    print "Formatting and cleaning dataset."
    # Features removal
    print "Removing features without name..."
    dataset.remove_feature_none()
    # Features replace
    print "Replacing media_color features with media_rgb..."
    t0 = time()
    dataset.replace_rgb_feature()
    print "Done. Time: {:.2f}s".format(time()-t0)
    print "Filling empty values with mean (contin) and 1st mode (categ)..."
    fill_empty_values(dataset)
    # Lookup and substitute categorical modes with 1 or 2 frequency
    print "Analysing especific features..."
    c = dataset.format_customer_values()
    print "Formated {0} values for 'customer'".format(c)
    categorical_features = ["caption_language", "hashtags_language",
        "location_country"]
    for feature in categorical_features:
        c = dataset.format_many_modes(feature)
        print "Formated {0} values for '{1}'".format(c, feature)
    # Replace outliers in continuous features
    print "Replacing outliers in some features..."
    cont_features = ['caption_char_lenght', 'caption_non_alpha_count',
        'caption_upper_count']
    for feature in cont_features:
        results = dataset.analyse_continuous_feature(feature)
        f_1q = results[6]
        f_3q = results[7]
        high_thre = f_3q + ((f_3q - f_1q) * 5)
        c = dataset.impute_feature_value(feature, high_thre,
                                         lambda val, t: val>t)
        if c>0:
            print "Imputed {0} values for feature {1}".format(c, feature)
    # The following features have outliers, remove them
    print "Removing samples with outliers..."
    cont_features = ['media_comments_count', 'media_likes_count']
    for feature in cont_features:
        results = dataset.analyse_continuous_feature(feature)
        f_1q = results[6]
        f_3q = results[7]
        high_thre = f_3q + ((f_3q - f_1q) * 5000)
        c = dataset.remove_samples(feature, high_thre, lambda val, t: val>t)
        if c>0:
            print "Discarded {0} samples due to feature {1} outlier.".format(
                c, feature)
    # Replace some features values with low importance
    print "Replacing low importance features values with a common value..."
    feature = "hashtags_category"
    new_value = "other"
    c = dataset.impute_feature_value(feature, new_value, 
                             lambda v, nv: any(word in v for word in ["hair", "holidays"]))
    if c>0:
        print "Imputed {0} values of '{1}' with '{2}'".format(c, feature,
            new_value)

    print "Preparing dataset and targets before entering algorithm..."
    t0 = time()
    dataset.split()
    dataset.transform_dataset()
    # Split in train and test arrays
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.get_dataset(), dataset.get_targets(), test_size=.30)
    print "DONE. Time preparing dataset: {:.2f}s".format(time()-t0)

    print "Modeling predictor... "
    predictor = Predictor(n_estimators=100)
    t0 = time()
    predictor.fit_algorithm(X_train, y_train)
    prediction = predictor.predict_outputs(X_train)
    t1 = time() - t0
    print "DONE. Time modeling predictor: {0:.2f}s.".format(t1)
    mse = predictor.predictor_metrics(y_train, prediction)
    print "MSE on train set: {0:.3f}".format(mse)

    print "Testing predictor... "
    t0 = time()
    score = predictor.score_predictor(X_test, y_test)
    prediction = predictor.predict_outputs(X_test)
    mse = predictor.predictor_metrics(y_test, prediction)
    t1 = time() - t0
    print "DONE. Time testing predictor: {0:.2f}s.".format(t1)
    print "Predictor score: {0:.3f}".format(score)
    print "MSE on test set: {0:.3f}".format(mse)
