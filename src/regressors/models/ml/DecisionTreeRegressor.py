
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

class RDecisionTree(object):

    def __init__(self, config):
        pass

    def train(self, features_train_set, target_train_set, features_validation_set):
        print("############features:",features_train_set.shape)
        print("############target:",target_train_set.shape)
        print("############target:",features_validation_set.shape)
        history = {}  # dictionary with all metrics
        mse = []
        val_mse = []  # mean squared error
        max_depth = range(2, 10, 1)
        for depth in max_depth:
            dtr = DecisionTreeRegressor(max_depth=depth)
            dtr.fit(features_train_set, target_train_set)
            # save the model in joblib file
            joblib.dump(dtr,
                        'decition_tree__regressor_h1_depth_{0}'.format(depth))
            evaluate_train = dtr.predict(features_train_set)
            print("hasta aquí")
            evaluate_validation = dtr.predict(features_validation_set)
            # metric calculation
            mse_train = mean_squared_error(target_train_set[['target']].values,
                                           evaluate_train)
            mse_validation = mean_absolute_error(
                target_validation_set[['target']].values,
                evaluate_validation)
            mse.append(mse_train)
            val_mse.append(mse_validation)
            # mean_squared_error()
            # mean_absolute_error()

        history['mse'] = mse
        history['val_mse'] = val_mse
