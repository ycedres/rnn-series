import sys
sys.path.insert(0, '/home/ycedres/Projects/PhD/wind/RNN-windPower/src/regressors/core/')

from model_runner import Experiment

from crossvalidation.csv_input_manager import CSVInputManager
from input_manager.supplychain_input_manager import SupplyChainInputManager
from output_manager.output_manager import OutputManager
from runner.runner import LocalRunner,TrainOperation,TestOperation
from config_manager.JSONConfigManager import JSONConfigManager

config_file_name = '/home/ycedres/Projects/PhD/wind/RNN-windPower/src/regressors/core/config_supplychain.json'
config_manager = JSONConfigManager(filename=config_file_name)

local_runner = LocalRunner()



basedir = config_manager.get_input_basedir()
input_filename = config_manager.get_input_filename()


for i in range(94):
    input_manager = SupplyChainInputManager()
    input_manager.configure_load_datasource(method='filesystem', colindex=i,
                                            filename=basedir+input_filename)   
    output_manager = OutputManager(expindex=i)                                             
    experiment = Experiment(config_manager=config_manager,
                                input_manager=input_manager,
                                output_manager=output_manager,
                                runner=local_runner)

    experiment.launch()