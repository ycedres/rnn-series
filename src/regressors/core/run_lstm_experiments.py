from model_runner import Experiment,FileConfigManager,LocalRunner, \
OutputManager,NRELInputManager,TrainOperation,TestOperation

from models.ml.RLSTM import RLSTM
from models.ml.RSimpleLSTM import RSimpleLSTM

import os
import pandas as pd

# CONFIGURATION MANAGER
config_file_name = '/home/ycedres/Projects/PhD/wind/RNN-windPower/src/regressors/core/config.ini'
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

# LAUNCH
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
    experiment.plot()

    experiment.save_error_estimators()


#######################################
# MAIN

if __name__ == "__main__":

    features = file_config_manager.get_features_config()
    # name = file_config_manager.get_file_prefix('lstm_1')
    name = 'lstm_3'

    lstm = RLSTM()
    # lstm = RSimpleLSTM()
    has_been_plotted = False

    for horizon in range(1,2):

        input_descriptor_string = 'ws'+features['window_size'] + '_' + \
                          'h'+str(horizon) + '_' + \
                          'p'+features['padding'] + '_' + \
                          'sz'+features['step_size'] + '_' + \
                          features['method']

        lstm.config_exp_path(basedir = file_config_manager.get_output_basedir(),
                    #file_prefix = file_config_manager.get_file_prefix(name),
                    file_prefix = name,
                    input_descriptor_string = input_descriptor_string)

        if not has_been_plotted:
            lstm.plot_model()

        output_filename = input_descriptor_string + '.csv'
        output_filename_dataframe = input_descriptor_string + '_dataframe' + '.csv'
        output_filename_path = file_config_manager.get_output_basedir() + output_filename
        output_filename_path_dataframe = file_config_manager.get_output_basedir() + output_filename_dataframe

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
            basedir = file_config_manager.get_output_basedir(),
            # file_prefix = file_config_manager.get_file_prefix(name),
            file_prefix = name,
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
