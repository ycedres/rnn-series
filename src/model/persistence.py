__author__ = 'tmorales'

import math
import numpy as np
import pandas as pd

from utils import Error
from verification.errors import ts_errors

# Nota: dentro del calculo del modelo solo np.array
#
def persistence(ts, window_size=20, horizont=1):

    # ts hat
    ts_hat = ts[window_size:]

    # persistence
    naive_ts = np.zeros(len(ts_hat), dtype=np.float)

    for i in range(0, len(naive_ts)):
        naive_ts[i] = ts[window_size + i - horizont]

    return naive_ts



def persistence2(ts,
                #ts_hat,
                #train_hat, test_hat, forecast_hat,
                window_size=20, horizon=1):

    ts_hat = ts[window_size:]

    # I have three time series: train, test, forecast but I need the
    # window_size previous to the time serie for the persistence

    # ts_hat
    naive_ts = np.zeros(len(ts_hat), dtype=np.float32)

    for i in range(0, len(ts_hat)):
        naive_ts[i] = ts[window_size + i - horizon]

    # ts_train --> ok
    #naive_train = np.zeros(len(train_hat))
    #for i in range(0, len(train_hat)):
    #    naive_train[i] = ts[window_size + i - horizon]

    # ts_test --> No
    #naive_forecast = np.zeros(len(forecast_hat), dtype=np.float32)
    #ts[len(train_hat):]

    #for i in range(0, len(forecast_hat)):
    #    pass


    # ts_test
    #naive_test = naive_ts[len(train_hat) : len(train_hat)+len(test_hat)]
    # ts_forecast
    #naive_forecast = naive_ts[len(train_hat)+len(test_hat) :
    #                          len(train_hat)+len(test_hat)+len(forecast_hat)]


    #print('h: {0} ts: {1} ts_train: {2}, ts_test: {3}, ts_forecast {4}'.format(horizon,
    #                                                                   naive_ts.shape,
    #                                                                   naive_train.shape,
    #                                                                   naive_test.shape,
    #                                                                  naive_forecast.shape))

    #ts_hatErrors  = ts_errors(ts_hat, naive_ts, horizon)
    #ts_trainErros = ts_errors(train_hat, naive_train, horizon)
    #ts_testErrors = ts_errors(test_hat, naive_test, horizon)
    #ts_forecastErrors   = ts_errors(forecast_hat, naive_forecast, horizon)

    return naive_ts
        #ts_hatErrors, ts_trainErros, ts_testErrors, ts_forecastErrors, \
        #   naive_train, naive_test, naive_forecast




def persistence3(_y_hat, _y_train, _y_test, window_size=20, horizon=1):
    """
    Persistence for test time serie

    :param _y_hat: todos los labels de la serie original
    :param _y_train: labels de la serie de entrenamiento
    :param _y_pred: labels de la serie test o forecast
    :param window_size: dimension de la ventana
    :param horizon: horizonte
    :return:
    """
    _y_hat  = _y_hat.values
    _y_test = _y_test['y_hat'].values


    naive_hat = np.zeros(len(_y_test), dtype=np.float32)

    for i in range(0, len(_y_test)):
        #
        # Nota: la persistencia la calculo con respecto a la
        #       medida enterior
        #
        naive_hat[i] = _y_hat[len(_y_train) + i - horizon]

    # Last point is the prediction of the last example = forecast
    naive_forecast = naive_hat[-1]


    # Tiene que ir en Verification
    # Errors between y_test and persistence
    # --- 1 order ---
    bias = Error.bias(_y_test, naive_hat)
    mae  = Error.mae(_y_test, naive_hat)
    mse  = Error.mse(_y_test, naive_hat)

    # --- 2 order ---
    rmse = Error.rmse(_y_test, naive_hat)
    sde  = Error.sde(_y_test, naive_hat)

    df_errors = pd.DataFrame({
                              #'r_2'  : [r_2],
                              'bias' : [bias],
                              'mae'  : [mae],
                              'mse'  : [mse],
                              'rmse' : [rmse],
                              'sde'  : [sde]
                             }, index=['h_{0}'.format(horizon)])

    return naive_hat, naive_forecast, df_errors