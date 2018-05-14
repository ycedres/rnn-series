import sys
sys.path.insert(0, '/home/ycedres/Projects/RNN/RNN-windPower/src/regressors/core/config_manager/')

from ConfigManager import ConfigManager
import json

class JSONConfigManager(ConfigManager):

    def __init__(self,filename):
        ConfigManager.__init__(self)
        self._filename = filename
        f = open(filename)
        self._config = json.load(f)

    def get_input_basedir(self):
        return self._config['data']['basedir']

    def get_output_basedir(self):
        return self._config['data']['output_basedir']

    def get_input_filename(self):
        return self._config['data']['filename']

    def get_file_prefix(self,name):
        return self._config['data']['file_prefix']

    def get_experiments(self):
        return self._config['experiments']

    def get_experiment_train_parameters(self,id):
        return self._config['experiments'][id]['train_parameters']

    #Returns {'train':{(opt1,val1),(opt2,val2)},'test':{(opt1,val1),(opt2,val2)}}
    def get_operation_config(self):
        pass

    def get_features_config(self):
        if self._features_config is None:
            return self._config['features']
        else:
            return self._features_config

    def get_horizon_range(self):
        return self._config["features"]["horizon_range"]

    #Returns {'knn':{(opt1,val1),(opt2,val2)},'lstm':{(opt1,val1),(opt2,val2)}}
    def get_model_config(self,name):
        return self._config['experiments'][name]["model_parameters"]

    def get_train_config(self,name):
        return self._config['experiments'][name]["train_parameters"]


    def get_experiment_description(self,name):
        return self._config['experiments'][name]['description']

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


if __name__ == "__main__":

    mgr = JSONConfigManager("/home/ycedres/Projects/RNN/RNN-windPower/src/regressors/core/config.json")
    #print(mgr.get_experiment_train_parameters(id="RLSTM001"))

    # [print(exp) for exp in mgr.get_experiments().items()]

    for expid,parameters in mgr.get_experiments().items():
        mgr.get_experiment_train_parameters(id=expid)
        print(mgr.get_features_config())
