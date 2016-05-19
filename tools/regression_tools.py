from time import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error
from print_tools import plot_with_bars

"""This file provides functions for the regression model of the data.
"""


def make_prediction(X_train, y_train, X_test, y_test, slice_samples=0,
                    pdetails=False):
    """This function receives training and testing inputs and outputs,
    performs a training using an sklearn algorithm and calculates the score
    using the testing inputs and outputs.
    """
    print 'Beginning prediction process...'

    regressor = RandomForestRegressor(bootstrap=True, n_jobs=-1,
                                      n_estimators=25)

    if slice_samples != 0:
        try:
            train_sl_len = len(X_train)/slice_samples
            test_sl_len = len(X_test)/slice_samples
        except TypeError:
            train_sl_len = len(X_train)
            test_sl_len = len(X_test)
        X_train = X_train[:train_sl_len]
        y_train = y_train[:train_sl_len]
        X_test = X_test[:test_sl_len]
        y_test = y_test[:test_sl_len]
    # else is not necessary, all the samples are taken

    t0 = time()
    regressor.fit(X_train, y_train)
    print "Time training algorithm: {0:.2f}s".format(time()-t0)

    if pdetails:
        print "Feature importances: {}".format(regressor.feature_importances_)
        print "Estimators: {}".format(regressor.estimators_)

    print 'Predicting results on test set ...'
    t0 = time()
    prediction = regressor.predict(X_test)
    print "Time prediction algorithm: {0:.2f}s".format(time()-t0)

    print "Cross validation score..."
    acc = mean_squared_error(y_test, prediction)
    acc1 = median_absolute_error(y_test, prediction)
    acc2 = regressor.score(X_test, y_test)

    print "Prediction accuracy and scores: \nmean_squared_error: {0},\
    \nmedian_absolute_error: {1},\
    \nregressor.score: {2}".format(acc, acc1, acc2)
    return regressor.feature_importances_


def outliers_cleaner(predictions, inputs, outputs, percentile=10):
    """Clean away the 10% of points that have the largest residual errors
    (difference between the prediction and the actual output).
    Return a list of tuples named cleaned_data where each tuple is of the
    form (x, y, error).
    """
    cleaned_data = []
    errors = []

    for pred_idx in range(0, len(predictions)):
        errors.append((abs(predictions[pred_idx]-outputs[pred_idx]),
                      pred_idx))

    errors.sort()
    # Discard the 10% values with bigger error
    idx_discard = len(errors)/percentile
    errors = errors[:-idx_discard]

    for error_idx in range(0, len(errors)):
        error, orig_idx = errors[error_idx]
        x_input = inputs[orig_idx]
        y_output = outputs[orig_idx]
        cleaned_data.append((x_input, y_output, error))
    return cleaned_data
