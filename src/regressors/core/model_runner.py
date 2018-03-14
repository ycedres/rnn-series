import sys
sys.path.insert(0, '../')

from crossvalidation.csv_input_manager import CSVInputManager
from input_manager.nrel_input_manager import NRELInputManager
from models.ml.KNNRegressor import KNNRegressor
from models.ml.DecisionTreeRegressor import RDecisionTree
from crossvalidation.train_test_split import TrainTestSplit
# from models.neural_networks.svr.SVR import SVRRegresion
import matplotlib.pyplot as plt
from configparser import ConfigParser
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

    def get_input_filename(self):
        return self._config.get('data', 'filename')

    #Returns {'train':{(opt1,val1),(opt2,val2)},'test':{(opt1,val1),(opt2,val2)}}
    def get_operation_config(self):
        pass

    def get_features_config(self):
        return dict(config.items('features'))

    #Returns {'knn':{(opt1,val1),(opt2,val2)},'lstm':{(opt1,val1),(opt2,val2)}}
    def get_model_config(self,name):
        config = {model[0]:self._config.items(model[0]) for model in self._config.items('models') if model[1]=='true'}

        return dict(config[name])

    def get_runner_config(self):
        pass

    def save_config():
        pass

    #[{model[0]:config.items(model[0])} for model in config.items('models') if model[1]=='true']
    #d={model[0]:config.items(model[0]) for model in config.items('models') if model[1]=='true'}

#/////////////////////////INPUT/OUTPUT/////////////////////////

#class InputManager():

class OutputManager(object):

    def __init__(self,output_file=''):
        pass

    def save(self,option='file'):
        if option == 'file':
            pass
        if option == 'print':
            print(self._data)

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

    # OUTPUT
    def save_output(self):
        self._output_manager.save(self._output)

    def print_output(self):
        self._output_manager.print_output(self._output)

    def print_performance(self):
        self._output_manager.print_performance()

    def plot(self):
        self._output_manager.plot_scatter(
                             self._output,
                             self._input_manager.get_test_target()
                             )


if __name__ == "__main__":

    config_file_name = '/home/ycedres/Projects/RNN/RNN-windPower/src/regressors/core/config.ini'
    file_config_manager = FileConfigManager(filename=config_file_name)
    filename = file_config_manager.get_input_filename()

    #input_manager = CSVInputManager(filename=filename, delimiter=';')
    output_manager = OutputManager()
    local_runner = LocalRunner()

    # experiment_knr = Experiment(config_manager=file_config_manager,
    #                             input_manager=input_manager,
    #                             output_manager=output_manager,
    #                             runner=local_runner)

    # Invoker
    # knr = KNNRegressor(file_config_manager.get_model_config(name='knr'))
    # knr_train_operation = TrainOperation(knr)
    # knr_test_operation = TestOperation(knr)
    #
    # experiment_knr.run_operation(knr_train_operation)
    # experiment_knr.run_operation(knr_test_operation)
    # experiment_knr.print_output()

    # DTR
    #df_input_manager = CSVInputManager(filename=filename, delimiter=';')

    input_manager = NRELInputManager()

    base_dir = '/home/ycedres/Projects/RNN/RNN-windPower/database/'
    filename = 'windpark_Offshore_WA_OR_turbine_25915.csv'
    input_manager.configure_load_datasource(method='filesystem',
    filename=base_dir+filename)

    input_manager.configure_features_generator(
        window_size=10,
        horizon=12,
        padding=0,
        step_size=1,
        write_csv_file=True,
        output_csv_file='/tmp/output_csv_file.csv',
        #method='sequential',
        method='daily'
    )

    input_manager.load_and_split()

    experiment_dtr = Experiment(config_manager=file_config_manager,
                                input_manager=input_manager,
                                output_manager=output_manager,
                                runner=local_runner)

    dtr = RDecisionTree(file_config_manager.
                        get_model_config(name='dtr'))

    dtr_train_operation = TrainOperation(dtr)
    experiment_dtr.run_train_operation(dtr_train_operation)

    dtr_test_operation = TestOperation(dtr)
    result = experiment_dtr.run_test_operation(dtr_test_operation)
    #experiment_dtr.print_output()

    experiment_dtr.plot()
