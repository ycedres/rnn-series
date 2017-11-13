__author__ = 'tmorales'

import numpy as np
import pandas as pd

class Error(object):

    def __int__(self, X, Y):
        self.X = X
        self.Y = Y

    @staticmethod
    def bias(X, Y):
        '''
        X = y_true
        Y = y_pred
        '''
        # poner que X tiene que se un array
        return np.sum(X - Y)/len(X)

    @staticmethod
    def mae(X, Y):
        return np.sum(abs(X - Y)) / len(X)

    @staticmethod
    def mse(X, Y):
        return np.sum(np.power(X -Y, 2)) / len(X)

    @staticmethod
    def rmse(X, Y):
        return np.sqrt(ca(np.power((X-Y), 2))) / len(X)

    @staticmethod
    def sde(X, Y):
        #return math.sqrt(np.sum(np.power((BIAS - eBIAS), 2)))
        pass

    @staticmethod
    def smape(X, Y):
        return np.sum(abs(Y - X) / (abs(Y) + abs(X))*0.5) / len(X)

def ts_errors(ts, ts_pred, horizon):

    errors = pd.DataFrame({
                           'bias'  : Error.bias(ts, ts_pred),
                           'mae'   : Error.mae(ts, ts_pred),
                           'mse'   : Error.mse(ts, ts_pred),
                           'rmse'  : Error.rmse(ts, ts_pred),
                           #'sde'   : Error.sde(ts, ts_pred)
                          },
                          index = ['h_{0}'.format(horizon)])

    return errors

def r2_score(persisitence, model):
    return
