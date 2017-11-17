
# -- scikit learn
from sklearn.neighbors import KNeighborsRegressor

from models.ml.fit_model import fit_models
from models.ml.predict_model import predict_models


class KNearestRegression(object):

    def __init__(self, config, exp_folder, data):
        self.config = config
        self.exp_folder = exp_folder
        self.data = data

        # instance the class
        n_neighbors = int(self.config.get('knr', 'n_neighbors'))
        weight = self.config.get('knr', 'weight')

        self.knr = KNeighborsRegressor(n_neighbors=n_neighbors,
                                       weights=weight)

    def fit(self):
        print('fitting k-nearest regression')
        fitted = fit_models(self.knr, self.data)

        return fitted

    def fit_predict(self):
        print('fit and predict k-nearest regression')
        fitted_model = fit_models(self.knr, self.data)
        predicted = predict_models(fitted_model, self.data)

        return predicted

    def predict(self):
        pass

    def run_knearest_regression(self):

        # action on the model
        action_on_model = self.config.get('knr', 'action_on_model')
        if action_on_model == 'fit':
            return self.fit()
        if action_on_model == 'fit_predict':
            return self.fit_predict()
        if action_on_model == 'predict':
            return self.predict()


class SupportVectorRegression(object):

    def __init__(self, config):
        self.config = config

    def fit(self) -> object:
        pass

    def fit_predict(self):
        pass

    def predict(self):
        pass

    def run_supportvector_regression(self):
        pass


class RandomForestRegression(object):

    def __init__(self, config):
        self.config = config

    def fit(self) -> object:
        pass

    def fit_predict(self):
        pass

    def predict(self):
        pass

    def run_randomforest_regression(self):
        pass


class XGBoostRegression(object):

    def __init__(self, config):
        self.config = config

    def fit(self) -> object:
        pass

    def fit_predict(self):
        pass

    def predict(self):
        pass

    def run_xgboost_regression(self):
        pass