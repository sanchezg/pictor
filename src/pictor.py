import sys
from time import time
from sklearn.cross_validation import train_test_split
from predictor import Predictor
from dataset import Dataset
from helper_functions import get_args_parsed, file_to_list


class Pictor(object):
    """
    Pictor is a class for create the Dataset and Predictor objects and perform
    the prediction for interactions target using the characteristics of the
    Predictor object.
    """
    def __init__(self, ds, pd):
        """Constructs the Pictor object with the dataset and predictor
        specified by arguments."""
        self.dataset = ds
        self.predictor = pd

    def clean_format_dataset(self):
        """Performs functions related to dataset analysis, formatting and
        cleaning."""
        print "Formatting and cleaning dataset..."
        self.dataset.split_dataset() # First to do
        self.dataset.replace_rgb_features()
        # Analyse continuous features
        for f in self.dataset.get_feature_names():
            if f in self.dataset.get_continuous_features():
                f_analysis = self.dataset.analyse_continuous_feature(f)
                if f_analysis is not None:
                    missing = f_analysis[0]
                    mean = f_analysis[1]
                    median = f_analysis[2]
                    std_dev = f_analysis[3]
                    f_max = f_analysis[4]
                    f_min = f_analysis[5]
                    f_1q = f_analysis[6]
                    f_3q = f_analysis[7]

                    # Impute missing values with mean
                    c = self.dataset.impute_feature_value(f, mean,
                                                lambda val, t: val==None)
                    print "Imputed {0} values for feature {1}".format(c, f)
            elif f in self.dataset.get_categorical_features():
                # Analyse categorical features
                f_analysis = self.dataset.analyse_categorical_feature(f)
                if f_analysis is not None:
                    valid_samples = f_analysis[0]
                    cardinality = f_analysis[1]
                    mode1 = f_analysis[2]
                    fmode1 = f_analysis[3]
                    mode2 = f_analysis[4]
                    fmode2 = f_analysis[5]

                    # Impute missing values with mean
                    c = self.dataset.impute_feature_value(f, mode1,
                                                lambda val, t: val==None)
                    print "Imputed {0} values for feature {1}".format(c, f)
            else:
                print "ERROR: '{0}' is not in any group".format(f)
        self.dataset.transform_dataset() # Last to do
        print "DONE."

    def prediction(self):
        """Performs functions related to predictor."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset.get_dataset(), self.dataset.get_targets(),
            test_size=.30)
        print "Modeling predictor... "
        t0 = time()
        self.predictor.fit_algorithm(X_train, y_train)
        prediction = self.predictor.predict_outputs(X_test)
        t1 = time() - t0
        print "DONE. Time modeling predictor: {0:.2f}s.".format(t1)
        mse = self.predictor.predictor_metrics(y_train, prediction)
        print "MSE on train set: {0:.3f}".format(mse)

        print "Testing predictor... "
        t0 = time()
        score = self.predictor.score_predictor(X_test, y_test)
        mse = self.predictor.predictor_metrics(y_test, prediction)
        t1 = time() - t0
        print "DONE. Time testing predictor: {0:.2f}s.".format(t1)
        print "Predictor score: {0:.3f}".format(score)
        print "MSE on test set: {0:.3f}".format(mse)

    def execute(self):
        """Executes the script purpose: load and work with dataset and make
        the prediction of targets."""
        self.clean_format_dataset()
        self.prediction()


welcome_msg = "Welcome to pictor: a predictor for pictures interactions.\n"

# Execute if this module is called directly
if __name__ == '__main__':
    print welcome_msg
    # Get program arguments
    csv_filename, features_filename = get_args_parsed()

    # Get features to discard from extern file
    if features_filename is not None:
        features_to_discard = file_to_list(features_filename)
    else:
        features_to_discard = None

    print "Loading dataset from file..."
    t0 = time()
    ds = Dataset(csv_filename, features_to_discard, empty_val=None)
    print "Done. Time loading dataset: {}".format(time()-t0)
    pd = Predictor(n_estimators=10)

    p = Pictor(ds, pd)
    p.execute() # Runs all the default steps for analysis and prediction
