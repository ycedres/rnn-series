
from models.neural_networks.mlp.topology import topology
from models.neural_networks.mlp.fit import fit
from models.neural_networks.mlp.predict import predict
from models.neural_networks.rnn.lstm_topology import lstm_topology
from models.neural_networks.rnn.stack_lstm_topology import stack_lstm_topology

class MLP(object):

    def __init__(self, config, exp_folder, data):
        self.config = config
        self.exp_folder = exp_folder
        self.data = data

    def fit(self):
        print('I am at MLP - fit')

        model = topology(self.config, self.exp_folder)
        model = fit(model, self.data)

        return model

    def fit_predict(self):
        model = self.fit()
        predicted = predict(model, self.data)

        return predicted

    def predict_load_model(self):
        pass

    def predict_load_weights(self):
        pass

    def auto(self):
        pass

    def run_mlp(self):

        # action on the models
        action_on_model = self.config.get('mlp', 'action_on_model')

        if action_on_model == 'fit':
            return self.fit()
        if action_on_model == 'fit_predict':
            return self.fit_predict()
        if action_on_model == 'predict_load_model':
            return self.predict_load_model()
        if action_on_model == 'predict_laod_weights':
            return self.predict_load_weights()
        if action_on_model == 'auto':
            return self.auto()

class RNN(object):

    def __init__(self, config, exp_folder):
        self.config = config
        self.exp_folder = exp_folder

    def fit(self) -> object:
        pass

    def fit_predict(self):
        pass

    def predict_laod_model(self):
        pass

    def predict_load_weights(self):
        pass

    def auto(self):
        pass

    def run_rnn(self):

        # action on the models
        action_on_model = self.config.get ('rnn', 'action_on_model')

        if action_on_model == 'fit':
            return self.fit()
        if action_on_model == 'fit_predict':
            return self.fit_predict()
        if action_on_model == 'predict_load_model':
            return self.predict_laod_model()
        if action_on_model == 'predict_laod_weights':
            return self.predict_load_weights()
        if action_on_model == 'auto':
            return self.auto()

class LSTM(object):

    def __init__(self, config, exp_folder):
        self.config = config
        self.exp_folder = exp_folder

    def fit(self):
        model = lstm_topology(self.config, self.exp_folder)

        return model

    def fit_predict(self):
        pass

    def predict_laod_model(self):
        pass

    def predict_load_weights(self):
        pass

    def auto(self):
        pass

    def run_lstm(self):

        # action on the models
        action_on_model = self.config.get ('lstm', 'action_on_model')

        if action_on_model == 'fit':
            return self.fit()
        if action_on_model == 'fit_predict':
            return self.fit_predict()
        if action_on_model == 'predict_load_model':
            return self.predict_laod_model()
        if action_on_model == 'predict_laod_weights':
            return self.predict_load_weights()
        if action_on_model == 'auto':
            return self.auto()

class StackLSTM(object):

    def __init__(self, config, exp_folder):
        self.config = config
        self.exp_folder = exp_folder

    def fit(self):
        model = stack_lstm_topology(self.config, self.exp_folder)

        return model

    def fit_predict(self):
        pass

    def predict_laod_model(self):
        pass

    def predict_load_weights(self):
        pass

    def auto(self):
        pass

    def run_stack_lstm(self):

        # action on the models
        action_on_model = self.config.get ('stack_lstm', 'action_on_model')

        if action_on_model == 'fit':
            return self.fit()
        if action_on_model == 'fit_predict':
            return self.fit_predict()
        if action_on_model == 'predict_load_model':
            return self.predict_laod_model()
        if action_on_model == 'predict_laod_weights':
            return self.predict_load_weights()
        if action_on_model == 'auto':
            return self.auto()


class GRU(object):
    pass