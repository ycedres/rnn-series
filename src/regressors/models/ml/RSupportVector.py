
from sklearn.svm import SVR

class RSupportVector(object):

    def __init__(self, config=None):
        self._reg = SVR(C=9,gamma='auto',kernel='rbf')

    def train(self, features_train_set,
                    target_train_set,
                    features_validation_set,
                    target_validation_set):
        self._reg.fit(features_train_set, target_train_set)

    def test(self,features_test_set):
        return self._reg.predict(features_test_set)
