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

    def __init__(self):
        self._model_config = None
        self._features_config = None
        self._data_config = None

    def set_features_config(self,d):
        self._features_config = d

    def set_model_config(self,d):
        self._model_config = d

    def set_data_config(self,d):
        self._data_config = d

class FileConfigManager(ConfigManager):

    def __init__(self,filename):
        ConfigManager.__init__(self)
        self._filename = filename
        self._config = ConfigParser()
        self._config.optionxform=str
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
        if self._features_config is None:
            return dict(self._config.items('features'))
        else:
            return self._features_config

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

    def set_output_config(self,save,basedir=None,file_prefix=None,
                          input_descriptor_string=None,
                          output_filename=None):
       self._save = save
       self._basedir = basedir
       self._file_prefix = file_prefix
       self._output_filename = output_filename
       self._input_descriptor_string = input_descriptor_string

    def print_output(self,data):
        print(data)

    def plot_scatter(self):
        x = self._df_prediction['target']
        y = self._df_prediction['prediction']

        plt.xlabel('prediction')
        plt.ylabel('target')
        plt.scatter(x,y,c='b')

        if self._save:

            directory = self._basedir + '/' + self._file_prefix + '/'
            filename = directory + 'scatter.png'

            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(filename)
        else:
            plt.show(block=False)

    def plot_scatter_diagonal(self,title):
        # df_x = pd.DataFrame(x)
        # df_x.index = y.index
        # df = pd.concat([df_x,y],axis=1)
        # df.columns = ['a','b']
        x = self._df_prediction['target']
        y = self._df_prediction['prediction']

        f, ax = plt.subplots(1,1,figsize=(10,10))
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        ax.set_xlim(x_min+1, x_max+1)
        ax.set_ylim(x_min+1, x_max+1)
        ax.plot((x_min, x_max), (x_min, x_max), lw=3, c='r')
        ax.scatter(x,y,c='b')
        ax.set(xlabel='target', ylabel='prediction')
        ax.set_title(title)
        #plt.figure()
        # self._df_prediction.plot(ax=ax,
        #         x='prediction',
        #         y='target',
        #         kind='scatter',
        #         c='b'
        #         )

        if self._save:

            directory = self._basedir + '/' + self._file_prefix + '_' + \
            self._input_descriptor_string + '/'

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

            directory = self._basedir + '/' + self._file_prefix + '_' + \
            self._input_descriptor_string + '/'
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

    def get_test_rmse(self):
        return np.sqrt(self.get_test_mse())

    def get_test_r2(self,reference_mse):
        return (reference_mse - self.get_test_mse()) / reference_mse

    def save_error_estimators(self,d):
        directory = self._basedir + '/' + self._file_prefix + '_' + \
        self._input_descriptor_string + '/'
        filename = directory + 'errors.json'

        if not os.path.exists(directory):

            os.makedirs(directory)

        f = open(filename,'w')
        import json
        d = json.dumps(d)
        f.write(d)
        f.close()


#/////////////////////////////////////////////////////////////

#Client
class Experiment(object):

    def __init__(self,config_manager,input_manager,output_manager,runner,description):
        self._config_manager = config_manager
        self._input_manager = input_manager
        self._output_manager = output_manager
        self._runner = runner
        self._data = None
        self._output = None
        self._description=description

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
        reference_mse = self._input_manager.get_reference_rmse()
        return {'mse':self._output_manager.get_test_mse(),
                'rmse':self._output_manager.get_test_rmse(),
                'mae':self._output_manager.get_test_mae(),
                'r2':self._output_manager.get_test_r2(reference_mse)}

    def save_error_estimators(self):
        self._output_manager.save_error_estimators(self.get_error_estimators())

    # OUTPUT
    def save_output(self):
        self._output_manager.save(self._output)

    def print_output(self):
        self._output_manager.print_output(self._output)

    def print_performance(self):
        self._output_manager.print_performance()

    def plot(self,type=None):
        if type=='scatter':
            # self._output_manager.plot_scatter()
            self._output_manager.plot_scatter_diagonal(title=self._description)
        else:
            self._output_manager.plot(
                                 self._output,
                                 self._input_manager.get_test_target()
            )

#/////////////////////////////////////////////////////////////

if __name__ == "__main__":

    def launch(regressor,config_manager,input_manager,output_manager,runner,
    description):

        experiment = Experiment(config_manager=file_config_manager,
                                    input_manager=input_manager,
                                    output_manager=output_manager,
                                    runner=local_runner,
                                    description=description)

        train_operation = TrainOperation(regressor)
        experiment.run_train_operation(train_operation)

        test_operation = TestOperation(regressor)
        experiment.run_test_operation(test_operation)

        experiment.plot(type='scatter')
        # experiment.plot()

        experiment.save_error_estimators()


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

    input_descriptor_string = 'ws'+features['window_size'] + '_' + \
                      'h'+features['horizon'] + '_' + \
                      'p'+features['padding'] + '_' + \
                      'sz'+features['step_size'] + '_' + \
                      features['method']

    output_filename = input_descriptor_string + '.csv'

    input_manager.configure_features_generator(
        window_size=int(features['window_size']),
        horizon=int(features['horizon']),
        padding=int(features['padding']),
        step_size=int(features['step_size']),
        write_csv_file=False,
        output_csv_file=file_config_manager.get_output_basedir()+
                        output_filename,
        #method='sequential',
        method=features['method']
    )

    input_manager.load_and_split()

    # EXPERIMENTOS

    #SVR
    # name = 'svr'
    # print(file_config_manager.get_model_config(name))
    # svr = RSupportVector(file_config_manager.get_model_config(name))
    #
    # output_manager.set_output_config(
    #     save = True,
    #     basedir = file_config_manager.get_output_basedir(),
    #     file_prefix = file_config_manager.get_file_prefix(name),
    #     input_descriptor_string = input_descriptor_string,
    #     output_filename = output_filename
    # )
    #
    # launch(
    #      regressor=svr,
    #      config_manager=file_config_manager,
    #      input_manager=input_manager,
    #      output_manager=output_manager,
    #      runner=local_runner)
    #
    # # Escribir la configuración en el directorio de salida
    # directory = file_config_manager.get_output_basedir() + '/' + \
    #             name + '_'  + input_descriptor_string + '/'
    # filename = 'config_' + name + '.json'
    # file_config_manager.write_cfg_file(directory+filename,name)


    #LSTM

    lstm = RLSTM()
    name = 'lstm'

    output_manager.set_output_config(
        save = True,
        basedir = file_config_manager.get_output_basedir(),
        file_prefix = file_config_manager.get_file_prefix(name),
        input_descriptor_string = input_descriptor_string,
        output_filename = output_filename
    )

    launch(
         regressor=lstm,
         config_manager=file_config_manager,
         input_manager=input_manager,
         output_manager=output_manager,
         runner=local_runner,
         description=input_descriptor_string)
    #
    # # Escribir la configuración en el directorio de salida
    # directory = file_config_manager.get_output_basedir() + '/' + \
    #             file_config_manager.get_file_prefix(name) + '/'
    # filename = 'config_' + name + '.json'
    # file_config_manager.write_cfg_file(directory+filename,name)


    # Esto es para que se muestren los gráficos en modo no bloqueante
    # plt.show()
    # import sys
    # sys.exit()
