
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
