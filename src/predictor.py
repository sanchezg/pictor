from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error


class Predictor(object):
    def __init__(self, n_estimators=50):
        """Constructor for the predictor object."""
        self.score = -1
        self.regressor = GradientBoostingRegressor(n_estimators=n_estimators)

    def fit_algorithm(self, x, y):
        """Wrapper to the sklearn regressor fit function."""
        self.regressor.fit(x, y)

    def predict_outputs(self, inputs):
        """Wrapper to the sklearn regressor predict function."""
        try:
            a = self.regressor.feature_importances_
        except sklearn.utils.validation.NotFittedError:
            print "Please fit the algorithm before calling this function."
            return
        prediction = self.regressor.predict(inputs)
        return prediction

    def predictor_metrics(self, outputs, prediction):
        """This function calculates metrics byt measuring MSE."""
        try:
            a = self.regressor.feature_importances_
        except sklearn.utils.validation.NotFittedError:
            print "Please fit the algorithm before calling this function."
            return
        return mean_squared_error(outputs, prediction)

    def score_predictor(self, x, y):
        """Wrapper to the score calculated with the predictor."""
        try:
            a = self.regressor.feature_importances_
        except sklearn.utils.validation.NotFittedError:
            print "Please fit the algorithm before calling this function."
            return
        return self.regressor.score(x, y)

if __name__ == '__main__':
    print "Please do not call this file directly."
