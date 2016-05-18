from time import time

"""This file provides functions for the regression model of the data.
"""


def make_prediction(X_train, y_train, X_test, y_test, sv=False,
                    slice_samples=0, pdetails=False):
    """This function receives training and testing inputs and outputs,
    performs a training using an sklearn algorithm and calculates the score
    using the testing inputs and outputs.
    """
    # from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, median_absolute_error

    print 'Beginning prediction process...'

    regressor = RandomForestRegressor(bootstrap=True, n_jobs=-1,
                                      n_estimators=25)
    # regr_parameters = {'n_estimators': [5, 10, 20], 
    #                    'max_features': [10, 25, "auto"],
    #                    'bootstrap': (True, False)}
    # regressor = GridSearchCV(regr, regr_parameters)

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

    if sv:
        t0 = time()

    regressor.fit(X_train, y_train)

    if sv:
        t1 = time() - t0
        print 'Time training algorithm: {}'.format(t1)

    if pdetails:
        # print regressor.best_estimator_
        print "Feature importances: {}".format(regressor.feature_importances_)
        print "Estimators: {}".format(regressor.estimators_)

    print 'Predicting results on test set ...'

    if sv:
        t0 = time()

    prediction = regressor.predict(X_test)

    if sv:
        t1 = time() - t0
        print 'Time prediction algorithm: {}'.format(t1)

    print "Cross validation score..."
    acc = mean_squared_error(y_test, prediction)
    acc1 = median_absolute_error(y_test, prediction)
    acc2 = regressor.score(X_test, y_test)

    if sv:
        print "Time testing algorithm: {0}.\
        \nAccuracy:\
        \nmean_squared_error: {1},\
        \nmedian_absolute_error: {2},\
        \nregressor.score: {3}".format(t1, acc, acc1, acc2)

    return


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
