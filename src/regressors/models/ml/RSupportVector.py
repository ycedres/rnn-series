
from sklearn.svm import SVR

class RSupportVector(object):

    def __init__(self, config=None,basedir=None,file_prefix=None,
                 input_descriptor_string=None):

        if basedir is not None:
            self._basedir = basedir
        if file_prefix is not None:
            self._file_prefix = file_prefix
        if input_descriptor_string is not None:
            self._input_descriptor_string = input_descriptor_string

        # self._reg = SVR(C=int(config_dict['C']),gamma=config_dict['gamma'],
        self._reg = SVR()

    def train(self, features_train_set,
                    target_train_set,
                    features_validation_set,
                    target_validation_set):
        self._reg.fit(features_train_set, target_train_set)

    def test(self,features_test_set):
        return self._reg.predict(features_test_set)


    def config_exp_path(self,basedir=None,file_prefix=None,
                 input_descriptor_string=None):

        self._basedir = basedir
        self._file_prefix = file_prefix
        self._input_descriptor_string = input_descriptor_string

    def plot_model(self):
        pass
