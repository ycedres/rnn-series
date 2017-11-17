

from configparser import ConfigParser

# -- models
from models.ml.run_models import KNearestRegression
from models.neural_networks.run_models import MLP
from models.neural_networks.run_models import LSTM
from models.neural_networks.run_models import StackLSTM

# -- preprocess
from testing.testing_fit import get_data

# -- utils
from utils import create_expFolder

class SelectModels(object):

    def __init__(self, config):
        self.config = config

        self.dic_models = {}
        keys = self.config.options('models')
        for key in keys:
            try:
                self.dic_models[key] = self.config.get('models', key)
                if self.dic_models[key] == -1:
                    pass
            except:
                print("exception on %s!" % keys)
                self.dic_models[key] = None

        # -- get data
        filename = self.config.get('data', 'filename')

        self.data, self.target_table = get_data(filename)

        # -- create experiment folder
        self.exp_folder = create_expFolder(config)

    def models(self):

        print('* I am at models')

        # -- machine learning
        # k-nearest regression
        if self.dic_models['knr'] == 'true':
            knr_obj = KNearestRegression(self.config, self.exp_folder, self.data)
            predicted = knr_obj.run_knearest_regression()

            self.target_table['predicted'] = predicted

            return self.target_table

        # support vector regression
        if self.dic_models['svr'] == 'true':
            pass

        # random forest regression
        if self.dic_models['rfr'] == 'true':
            pass

        # xgboost regression
        if self.dic_models['xgboostr'] == 'true':
            pass

        # -- neural network
        # fully-connect neural network regression
        if self.dic_models['mlp'] == 'true':
            print('I am in models - MLP')
            mlp_obj = MLP(self.config, self.exp_folder, self.data)
            predicted = mlp_obj.run_mlp()

            self.target_table['predicted'] = predicted

            return self.target_table

        # recurrent neural network regression
        if self.dic_models['rnn'] == 'true':
            pass

        # long short-term memory regression
        if self.dic_models['lstm'] == 'true':

            print('I am in model - LSTM')
            lstm_obj = LSTM(self.config, self.exp_folder)
            predicted = lstm_obj.run_lstm()

            return predicted

        # long short-term memory regression
        if self.dic_models['stack_lstm'] == 'true':

            print ('I am in model - Stack LSTM')
            lstm_obj = StackLSTM(self.config, self.exp_folder)
            predicted = lstm_obj.run_stack_lstm()

            return predicted


if __name__ == "__main__":

    print('Runinig .. ')
    config = ConfigParser()
    config.read('config.ini')

    select = SelectModels(config)

    table_target = select.models()

    print(table_target)


    print('Stop')