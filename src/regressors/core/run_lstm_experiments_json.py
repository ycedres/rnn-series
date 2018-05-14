import sys
sys.path.insert(0, '/home/ycedres/Projects/RNN/RNN-windPower/src/regressors/core/')

from model_runner import Experiment,LocalRunner, \
OutputManager,NRELInputManager,TrainOperation,TestOperation

from config_manager.JSONConfigManager import JSONConfigManager

import os
import importlib
import pandas as pd

# CONFIGURATION MANAGER
config_file_name = '/home/ycedres/Projects/RNN/RNN-windPower/src/regressors/core/config.json'
config_manager = JSONConfigManager(filename=config_file_name)
basedir = config_manager.get_input_basedir()
filename = config_manager.get_input_filename()

# OUTPUT MANAGER
output_manager = OutputManager()
# RUNNER
local_runner = LocalRunner()

# INPUT MANAGER
input_manager = NRELInputManager()
input_manager.configure_load_datasource(method='filesystem',
filename=basedir+filename)

# LAUNCH
def launch(regressor,config_manager,input_manager,output_manager,runner,
description):

    experiment = Experiment(config_manager=config_manager,
                                input_manager=input_manager,
                                output_manager=output_manager,
                                runner=local_runner,
                                description=description)

    train_operation = TrainOperation(regressor)
    experiment.run_train_operation(train_operation)

    test_operation = TestOperation(regressor)
    experiment.run_test_operation(test_operation)

    experiment.plot(type='scatter')
    experiment.plot()

    experiment.save_error_estimators()

    experiment.save_experiment_descriptor(
        experiment_name=type(regressor).__name__,
        features_config=config_manager.get_features_config(),
        train_config=config_manager.get_train_config(name=type(regressor).__name__),
        model_config=config_manager.get_model_config(name=type(regressor).__name__),
        description=config_manager.get_experiment_description(name=type(regressor).__name__)
    )


#######################################
# MAIN

if __name__ == "__main__":

    features = config_manager.get_features_config()
    print(features)
    # name = file_config_manager.get_file_prefix('lstm_1')
    # name = 'lstm_3'

    # lstm = RLSTM()
    # lstm = RSimpleLSTM()
    # lstm = RStackedLSTM()
    has_been_plotted = False
    horizon_range = config_manager.get_horizon_range()

    for expid,parameters in config_manager.get_experiments().items():

        for horizon in (horizon_range["start"],horizon_range["end"]):

            MClass = getattr(importlib.import_module("models.ml."+expid),expid)
            model = MClass()
            model.set_batch_size(1024)
            model.set_epochs(10)

            input_descriptor_string = 'ws'+str(features['window_size']) + '_' + \
                              'h'+str(horizon) + '_' + \
                              'p'+str(features['padding']) + '_' + \
                              'sz'+str(features['step_size']) + '_' + \
                              features['method']

            model.config_exp_path(basedir = config_manager.get_output_basedir(),
                        #file_prefix = file_config_manager.get_file_prefix(name),
                        file_prefix = expid,
                        input_descriptor_string = input_descriptor_string)

            if not has_been_plotted:
                model.plot_model()

            output_filename = input_descriptor_string + '.csv'
            output_filename_dataframe = input_descriptor_string + '_dataframe' + '.csv'
            output_filename_path = config_manager.get_output_basedir() + output_filename
            output_filename_path_dataframe = config_manager.get_output_basedir() + output_filename_dataframe

            input_manager.configure_features_generator(
                window_size=int(features['window_size']),
                horizon=horizon,
                padding=int(features['padding']),
                step_size=int(features['step_size']),
                write_csv_file=True,
                output_csv_file=output_filename_path_dataframe,
                #method='sequential',
                method=features['method']
            )



            if os.path.exists(output_filename_path_dataframe):
                parse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
                df = pd.read_csv(output_filename_path_dataframe,sep=';',date_parser=parse,index_col=0)
                input_manager.load_dataframe(df)
            else:
                input_manager.load_and_split()

            output_manager.set_output_config(
                save = True,
                basedir = config_manager.get_output_basedir(),
                # file_prefix = file_config_manager.get_file_prefix(name),
                file_prefix = expid,
                input_descriptor_string = input_descriptor_string,
                output_filename = output_filename
            )

            launch(
                 regressor=model,
                 config_manager=config_manager,
                 input_manager=input_manager,
                 output_manager=output_manager,
                 runner=local_runner,
                 description=input_descriptor_string)
