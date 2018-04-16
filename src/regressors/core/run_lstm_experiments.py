from model_runner import Experiment,FileConfigManager,LocalRunner, \
OutputManager,NRELInputManager,TrainOperation,TestOperation

from models.ml.RLSTM import RLSTM
from models.ml.RSimpleLSTM import RSimpleLSTM

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
    lstm = RLSTM()
    #lstm = RSimpleLSTM()
    name = 'lstm'

    for horizon in range(1,2):

        input_descriptor_string = 'ws'+features['window_size'] + '_' + \
                          'h'+str(horizon) + '_' + \
                          'p'+features['padding'] + '_' + \
                          'sz'+features['step_size'] + '_' + \
                          features['method']

        output_filename = input_descriptor_string + '.csv'

        input_manager.configure_features_generator(
            window_size=int(features['window_size']),
            horizon=horizon,
            padding=int(features['padding']),
            step_size=int(features['step_size']),
            write_csv_file=True,
            output_csv_file=file_config_manager.get_output_basedir()+
                            output_filename,
            #method='sequential',
            method=features['method']
        )

        input_manager.load_and_split()

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
