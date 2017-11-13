__author__ = 'tmorales'

"""
implementada ... probarla despues de SVR


"""
import time
import numpy as np
import pandas as pd
from verification.errors import ts_errors

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

from utils import Error



def fit_model(model, _X_train, _y_train, _X_test, _y_test,
              tuned_parameters=None,
              number_KFold=10,
              horizon=1):
    """

    :param model:
    :param _X_train:
    :param _y_train:
    :param _X_test:
    :param _y_test:
    :param tuned_parameters:
    :param cv_method:
    :return:
    """
    # From Pandas DF --> np.array
    #_y_train = _y_train['y_hat'].values
    #_y_test = _y_test['y_hat'].values

    # KFold
    cv_method = KFold(len(_y_train), number_KFold)

    grid = GridSearchCV(model,
                        param_grid=tuned_parameters,
                        cv=cv_method,
                        verbose=0)
    # Fit the model
    grid_result = grid.fit(_X_train, _y_train)

    best_model = grid_result.best_estimator_

    # pred train data
    pred_train = best_model.predict(_X_train)

    # pred test data
    pred_test  = best_model.predict(_X_test)

    # pred forecast
    pred_forecast = pred_test[-1]

    # error of training data
    # ------------------------
    df_trainErrors = ts_errors(pred_train, _y_train, horizon)
    df_testErrors  = ts_errors(pred_test, _y_test, horizon)


    return best_model, pred_test, df_trainErrors, df_testErrors