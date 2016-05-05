def slice_begin_end(array_len, percentile):
    from random import randint, seed
    from time import clock
    seed(int(clock()))

    last_begin = array_len / percentile
    final_len = array_len * (1 - 1/percentile)

    begin = randint(0, last_begin)
    end = begin + final_len
    while end >= array_len:
        begin = randint(0, last_begin)
        end = begin + final_len
    return begin, end


def make_prediction(X_train, y_train, X_test, y_test, show_score=False,
                    slice_samples=0):
    """This function receives training and testing inputs and outputs,
    performs a training using an sklearn algorithm and calculates the score
    using the testing inputs and outputs.
    """
    # from sklearn.svm import SVR
    from sklearn.svm import NuSVR
    # from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.linear_model import SGDRegressor
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.linear_model import LinearRegression
    # from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import mean_squared_error, median_absolute_error
    from time import time

    # regressor = SVR()
    regressor = NuSVR()
    # regressor = KNeighborsRegressor()
    # regressor = SGDRegressor()
    # regressor = LogisticRegression(solver='sag', max_iter=100, n_jobs=2)
    # regressor = LinearRegression()
    # regressor = KernelRidge()

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
    t1 = time() - t0
    print 'Time training algorithm: {}'.format(t1)

    # clean outliers
    print 'Finding outliers...'
    t0 = time()
    prediction = regressor.predict(X_train)
    cleaned_data = outliers_cleaner(prediction, X_train, y_train)
    t1 = time() - t0
    print 'Time finding outliers: {}'.format(t1)
    if len(cleaned_data) > 0:
        print 'Re-fitting algorithm due to outliers detection...'
        t0 = time()
        x_input, y_output, errors = zip(*cleaned_data)
        regressor.fit(x_input, y_output)  # re-fit
        t1 = time() - t0
        print 'Time re-fitting algorithm: {}'.format(t1)

    print 'Predicting results on test set ...'
    t0 = time()
    prediction = regressor.predict(X_test)
    t1 = time() - t0
    print 'Time prediction algorithm: {}'.format(t1)

    print 'Calculates score and accuracy...'
    t0 = time()
    acc = mean_squared_error(y_test, prediction)
    acc1 = median_absolute_error(y_test, prediction)
    acc2 = regressor.score(X_test, y_test)
    t1 = time() - t0
    print "Time testing algorithm: {0}.\
    \nAccuracy:\
    \nmean_squared_error: {1},\
    \nmedian_absolute_error: {2},\
    \nregressor.score: {3}".format(t1, acc, acc1, acc2)


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
