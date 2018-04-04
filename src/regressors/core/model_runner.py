import sys
sys.path.insert(0, '../')

from crossvalidation.csv_input_manager import CSVInputManager
from input_manager.nrel_input_manager import NRELInputManager
from models.ml.KNNRegressor import KNNRegressor
from models.ml.DecisionTreeRegressor import RDecisionTree
from models.ml.RSupportVector import RSupportVector
from models.ml.RLSTM import RLSTM
from crossvalidation.train_test_split import TrainTestSplit
# from models.neural_networks.svr.SVR import SVRRegresion
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from configparser import ConfigParser
import pandas as pd
import numpy as np
import os
# Models == Receivers



# Invoker
class Runner(object):
    def __init__(self):
        self._queue = []

    #def history():
    #    return self._history

    def local_run(operation): #la operacion tiene al modelo como atributo
        #self._history = self._history + (operation,)
        #self._queue.append(operation)
        operation.execute()

class LocalRunner(Runner):
    def __init__(self):
        self._queue = []

    def append(operation):
        self._queue.append(operation)

    def run_queue(operation):
        pass

    def run_operation(operation):
        operation.excecute()

# Command


class Operation(object):
    def __init__(self, model):
        self._model = model

# Specific Commands


class TrainOperation(Operation):

    def __init__(self, model):
        self._model = model

    def run(self, train_features,
                  train_target,
                  validation_features,
                  validation_target):

        self._model.train(train_features, train_target, validation_features,
                          validation_target)


class TestOperation(Operation):

    def __init__(self, model):
        self._model = model

    def run(self, test_features):
        return self._model.test(test_features)



#/////////////////////////CONFIG MANAGER/////////////////////////
#//Configuration Manager

class ConfigManager:

    def __init__(self,type):
        pass



class FileConfigManager(ConfigManager):

    def __init__(self,filename):
        self._filename = filename
        self._config = ConfigParser()
        self._config.read(filename)

    def get_input_basedir(self):
        return self._config.get('data', 'basedir')

    def get_output_basedir(self):
        return self._config.get('data', 'output_basedir')

    def get_input_filename(self):
        return self._config.get('data', 'filename')

    def get_file_prefix(self,name):
        return self._config.get(name,'file_prefix')

    #Returns {'train':{(opt1,val1),(opt2,val2)},'test':{(opt1,val1),(opt2,val2)}}
    def get_operation_config(self):
        pass

    def get_features_config(self):
        return dict(self._config.items('features'))

    #Returns {'knn':{(opt1,val1),(opt2,val2)},'lstm':{(opt1,val1),(opt2,val2)}}
    def get_model_config(self,name):
        config = {model[0]:self._config.items(model[0]) for model in self._config.items('models') if model[1]=='true'}

        return dict(config[name])

    def get_runner_config(self):
        pass

    def save_config():
        pass

    def write_cfg_file(self,filename,name):
        d = self.get_model_config(name)
        f = open(filename,'w')
        import json
        d = json.dumps(d)
        f.write(d)
        f.close()

    #[{model[0]:config.items(model[0])} for model in config.items('models') if model[1]=='true']
    #d={model[0]:config.items(model[0]) for model in config.items('models') if model[1]=='true'}

#/////////////////////////INPUT/OUTPUT/////////////////////////

#class InputManager():

class OutputManager(object):

    def __init__(self,output_file=''):
        self._df_prediction = None
        pass

    def save(self,option='file'):
        if option == 'file':
            pass
        if option == 'print':
            print(self._data)

    def set_output_config(self,save,basedir,file_prefix,output_filename):
       self._save = save
       self._basedir = basedir
       self._file_prefix = file_prefix
       self._output_filename = output_filename

    def print_output(self,data):
        print(data)

    def plot_scatter(self,x,y):
        f, ax = plt.subplots(1,1,figsize=(10,10))
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        ax.set_xlim(x_min+1, x_max+1)
        ax.set_ylim(x_min+1, x_max+1)
        ax.plot((x_min, x_max), (x_min, x_max), lw=3, c='r')
        ax.scatter(x,y)
        plt.show()

    def plot_scatter_df(self):
        # df_x = pd.DataFrame(x)
        # df_x.index = y.index
        # df = pd.concat([df_x,y],axis=1)
        # df.columns = ['a','b']
        x = self._df_prediction['prediction']
        y = self._df_prediction['target']

        f, ax = plt.subplots(1,1,figsize=(10,10))
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        ax.set_xlim(x_min+1, x_max+1)
        ax.set_ylim(x_min+1, x_max+1)
        ax.plot((x_min, x_max), (x_min, x_max), lw=3, c='r')
        ax.scatter(x,y)
        #plt.figure()
        self._df_prediction.plot(ax=ax,
                x='prediction',
                y='target',
                kind='scatter',
                )

        if self._save:

            directory = self._basedir + '/' + self._file_prefix + '/'
            filename = directory + 'scatter.png'

            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(filename)
        else:
            plt.show(block=False)



    def plot(self,x,y):
        x = pd.DataFrame(x)
        x.index = y.index
        df = pd.concat([x,y],axis=1)

        df.plot(figsize=(15,5))
        # plt.plot(x,y)
        if self._save:
            directory = self._basedir + '/' + self._file_prefix + '/'
            filename = directory + 'serie.png'

            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig(filename)
        else:
            plt.show(block=False)


    def set_df_prediction(self,x:np.array,y:pd.DataFrame):
        df_x = pd.DataFrame(x)
        df_x.index = y.index
        self._df_prediction = pd.concat([df_x,y],axis=1)
        self._df_prediction.columns = ['prediction','target']
        if self._save:

            directory = self._basedir + '/' + self._file_prefix + '/'
            filename = directory + self._output_filename

            if not os.path.exists(directory):
                os.makedirs(directory)

            self._df_prediction.to_csv(filename,sep=';')


    def get_test_mae(self):
        self._df_prediction['mae'] = np.abs(self._df_prediction.prediction -
                                      self._df_prediction.target)
        return  self._df_prediction.mae.sum() / len(self._df_prediction.mae)

    def get_test_mse(self):
        self._df_prediction['mse'] = np.power(self._df_prediction.prediction -
                                       self._df_prediction.target, 2)
        return self._df_prediction.mse.sum() / len(self._df_prediction.mse)






#/////////////////////////////////////////////////////////////

#Client
class Experiment(object):

    def __init__(self,config_manager,input_manager,output_manager,runner):
        self._config_manager = config_manager
        self._input_manager = input_manager
        self._output_manager = output_manager
        self._runner = runner
        self._data = None
        self._output = None

    # INPUT

    def get_features_target(self):
        return self._input_manager.get_features_target()

    def read_data(self):
        return self._input_manager.get_data()

    def read_input(self):
        return self._input_manager.read_input()

    def read_dataframe(self):
        return self._input_manager.read_dataframe()

    def read_input_df(self):
        return self._input_manager.load_data()

    def read_target(self):
        return self._input_manager.read_target()

    # RUNNER
    def run_train_operation(self, operation):
        # self._output = operation.run(self.read_input())
        operation.run(
             train_features=self._input_manager.get_train_features(),
             train_target=self._input_manager.get_train_target(),
             validation_features=self._input_manager.get_validation_features(),
             validation_target=self._input_manager.get_validation_target()
        )

    def run_test_operation(self,operation):
        self._output = operation.run(
            test_features=self._input_manager.get_test_features(),
        )
        self._output_manager.set_df_prediction(
                             self._output,
                             self._input_manager.get_test_target()
                             )

    def get_error_estimators(self):
        return {'mse':self._output_manager.get_test_mse(),
                'mae':self._output_manager.get_test_mae()}

    # OUTPUT
    def save_output(self):
        self._output_manager.save(self._output)

    def print_output(self):
        self._output_manager.print_output(self._output)

    def print_performance(self):
        self._output_manager.print_performance()

    def plot(self,type=None):
        if type=='scatter':
            self._output_manager.plot_scatter_df(

                                 )
        else:
            self._output_manager.plot(
                                 self._output,
                                 self._input_manager.get_test_target()
            )

#/////////////////////////////////////////////////////////////

if __name__ == "__main__":

    def exp_svr(config_manager,input_manager,output_manager,runner):

        experiment_svr = Experiment(config_manager=file_config_manager,
                                    input_manager=input_manager,
                                    output_manager=output_manager,
                                    runner=local_runner)

        svr = RSupportVector()
        svr_train_operation = TrainOperation(svr)
        experiment_svr.run_train_operation(svr_train_operation)

        svr_test_operation = TestOperation(svr)

        experiment_svr.run_test_operation(svr_test_operation)

        experiment_svr.plot(type='scatter')

        experiment_svr.plot()

        print(experiment_svr.get_error_estimators())

    def exp_lstm(config_manager,input_manager,output_manager,runner):
        experiment_lstm = Experiment(config_manager=file_config_manager,
                                    input_manager=input_manager,
                                    output_manager=output_manager,
                                    runner=local_runner)

        lstm = RLSTM()
        lstm_train_operation = TrainOperation(lstm)
        experiment_lstm.run_train_operation(lstm_train_operation)

        lstm_test_operation = TestOperation(lstm)

        experiment_lstm.run_test_operation(lstm_test_operation)

        experiment_lstm.plot(type='scatter')

        experiment_lstm.plot()

        print(experiment_lstm.get_error_estimators())

    def launch(regressor,config_manager,input_manager,output_manager,runner):
        experiment = Experiment(config_manager=file_config_manager,
                                    input_manager=input_manager,
                                    output_manager=output_manager,
                                    runner=local_runner)

        train_operation = TrainOperation(regressor)
        experiment.run_train_operation(train_operation)

        test_operation = TestOperation(regressor)
        experiment.run_test_operation(test_operation)

        experiment.plot(type='scatter')
        experiment.plot()





    # CONFIGURATION MANAGER
    config_file_name = '/home/ycedres/Projects/RNN/RNN-windPower/src/regressors/core/config.ini'

    file_config_manager = FileConfigManager(filename=config_file_name)
    basedir = file_config_manager.get_input_basedir()
    filename = file_config_manager.get_input_filename()

    # OUTPUT MANAGER
    output_manager = OutputManager()

    # RUNNER
    local_runner = LocalRunner()

    # INPUT MANAGER
    input_manager = NRELInputManager()
    input_manager.configure_load_datasource(method='filesystem',
    filename=basedir+filename)

    features = file_config_manager.get_features_config()

    # import sys
    # sys.exit()
    output_filename = 'ws'+features['window_size'] + '_' + \
                      'h'+features['horizon'] + '_' + \
                      'p'+features['padding'] + '_' + \
                      'sz'+features['step_size'] + '_' + \
                      features['method'] +\
                      '.csv'

    input_manager.configure_features_generator(
        window_size=int(features['window_size']),
        horizon=int(features['horizon']),
        padding=int(features['padding']),
        step_size=int(features['step_size']),
        write_csv_file=True,
        output_csv_file=file_config_manager.get_output_basedir()+
                        output_filename,
        #method='sequential',
        method=features['method']
    )

    input_manager.load_and_split()

    # EXPERIMENTOS

    #SVR

    svr = RSupportVector()

    output_manager.set_output_config(
        save = True,
        basedir = file_config_manager.get_output_basedir(),
        file_prefix = file_config_manager.get_file_prefix('svr'),
        output_filename = output_filename
    )

    launch(
         regressor=svr,
         config_manager=file_config_manager,
         input_manager=input_manager,
         output_manager=output_manager,
         runner=local_runner)


    directory = file_config_manager.get_output_basedir() + '/' + \
                file_config_manager.get_file_prefix('svr') + '/'
    filename = 'config_svr.json'
    file_config_manager.write_cfg_file(directory+filename,'svr')


    # Esto es para que se muestren los gráficos en modo no bloqueante
    # plt.show()
    # import sys
    # sys.exit()
