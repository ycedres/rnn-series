import sys
sys.path.insert(0, '../')

from crossvalidation.csv_input_manager import CSVInputManager
from input_manager.nrel_input_manager import NRELInputManager
from output_manager.output_manager import OutputManager
from runner.runner import LocalRunner,TrainOperation,TestOperation

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
import importlib

#Client
class Experiment(object):

    def __init__(self,config_manager,input_manager,output_manager,runner,description=None):
        self._config_manager = config_manager
        self._input_manager = input_manager
        self._output_manager = output_manager
        self._runner = runner
        self._data = None
        self._output = None
        self._description=description
        self._model = None

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

    def launch(self):
        config_manager = self._config_manager
        for expid,parameters in config_manager.get_experiments().items():
            train_parameters = config_manager.get_train_config(expid)
            model_parameters = config_manager.get_model_config(expid)
            description = config_manager.get_experiment_description(expid)
            self._launch_regressor(expid,train_parameters,model_parameters,description)

    def _launch_regressor(self,expid,train_parameters,model_parameters,description):

        MClass = getattr(importlib.import_module("models.ml."+expid),expid)
        self._model = MClass()
        #self._model.configure(model_parameters)
        self._model.configure_train(train_parameters)
        self._description = self._config_manager.get_experiment_description(expid)

        window_range = self._config_manager.get_window_range()

        horizon_range = self._config_manager.get_horizon_range()
        has_been_plotted = False
        features = self._config_manager.get_features_config()
        window_range = (window_range['start'],window_range['end'])
        horizon_range = (horizon_range['start'],horizon_range['end'])
        for ws in window_range:
            for hr in horizon_range:
                self._model.configure(features_by_timestep=ws)
                # Cadena que describe el formato de la entrada
                input_descriptor_string = 'ws'+str(ws) + '_' + \
                              'h'+str(hr) + '_' + \
                              'p'+str(features['padding']) + '_' + \
                              'sz'+str(features['step_size']) + '_' + \
                              features['method']

                # En caso de que el modelo necesite escribir algo en disco
                self._model.config_exp_path(basedir = self._config_manager.get_output_basedir(),
                            #file_prefix = file_config_manager.get_file_prefix(name),
                            file_prefix = expid,
                            input_descriptor_string = input_descriptor_string)

                if not has_been_plotted:
                    self._model.plot_model()

                output_filename = input_descriptor_string + '.csv'
                output_filename_dataframe = input_descriptor_string + '_dataframe' + '.csv'
                output_filename_path = self._config_manager.get_output_basedir() + output_filename
                output_filename_path_dataframe = self._config_manager.get_output_basedir() + output_filename_dataframe

                self._input_manager.configure_features_generator(
                    window_size=ws,
                    horizon=hr,
                    padding=int(features['padding']),
                    step_size=int(features['step_size']),
                    write_csv_file=True,
                    output_csv_file=output_filename_path_dataframe,
                    #method='sequential',
                    method=features['method']
                )

                if os.path.exists(output_filename_path_dataframe):
                    print("output_filename_path_dataframe: " + output_filename_path_dataframe + " @@@@@@@@@@@@@@@@@@@@@@ EXISTE")
                    parse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
                    df = pd.read_csv(output_filename_path_dataframe,sep=';',date_parser=parse,index_col=0)
                    self._input_manager.load_dataframe(df)
                else:
                    print("@@@@@@@@@@@@@@@@@@@@@@ NO EXISTE")
                    self._input_manager.load_and_split()

                self._output_manager.set_output_config(
                    save = True,
                    basedir = self._config_manager.get_output_basedir(),
                    # file_prefix = file_config_manager.get_file_prefix(name),
                    file_prefix = expid,
                    horizon=hr,
                    input_descriptor_string = input_descriptor_string,
                    output_filename = output_filename
                )

                train_operation = TrainOperation(self._model)
                self.run_train_operation(train_operation)

                test_operation = TestOperation(self._model)
                self.run_test_operation(test_operation)

                #experiment.generate_report()

                self.plot(type='scatter')
                self.plot()

                self.save_error_estimators()

                self.save_experiment_descriptor(
                    experiment_name=type(self._model).__name__,
                    features_config=self._config_manager.get_features_config(),
                    train_config=self._config_manager.get_train_config(name=type(self._model).__name__),
                    model_config=self._config_manager.get_model_config(name=type(self._model).__name__),
                    description=self._config_manager.get_experiment_description(name=type(self._model).__name__)
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

    def save_experiment_descriptor(self,experiment_name,features_config,train_config,
                                   model_config,description):
        self._output_manager.save_experiment_descriptor(
            experiment_name=experiment_name,
            features_config=features_config,
            train_config=train_config,
            model_config=model_config,
            errors=self.get_error_estimators(),
            description=description)


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
