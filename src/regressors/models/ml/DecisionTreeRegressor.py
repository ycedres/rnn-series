
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.metrics.regression import mean_squared_error
from sklearn.metrics.regression import mean_absolute_error

class RDecisionTree(object):

    def __init__(self, config):
        pass

    def train(self, features_train_set, target_train_set, features_validation_set,target_validation_set):
        history = {}  # dictionary with all metrics
        mse = []
        val_mse = []  # mean squared error
        max_depth = range(2, 10, 1)
        for depth in max_depth:
            self._dtr = DecisionTreeRegressor(max_depth=depth)
            self._dtr.fit(features_train_set, target_train_set)
            # save the model in joblib file
            joblib.dump(self._dtr,
                        'decition_tree__regressor_h1_depth_{0}'.format(depth))
            evaluate_train = self._dtr.predict(features_train_set)
            evaluate_validation = self._dtr.predict(features_validation_set)
            # metric calculation
            mse_train = mean_squared_error(target_train_set.values,
                                           evaluate_train)
            mse_validation = mean_absolute_error(
                target_validation_set.values,
                evaluate_validation)
            mse.append(mse_train)
            val_mse.append(mse_validation)
            # mean_squared_error()
            # mean_absolute_error()

        history['mse'] = mse
        history['val_mse'] = val_mse



    def test(self,features_test_set):
        return self._dtr.predict(features_test_set)
