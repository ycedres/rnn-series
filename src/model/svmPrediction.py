import time
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

from utils import Error



def svr(_X_train, _y_train, _X_test, _y_test,
        tuned_parameters=None, number_KFold =10,
        horizon=1):
    """

    :param _X_train: must be a np.array
    :param _y_train: must be a np.array
    :param _X_test: must be a np.array
    :param _y_test: must be a np.array
    :param tuned_parameters:
    :param cv_method:
    :return:
    """

    # From Pandas DF --> np.array
    _y_train = _y_train['y_hat'].values
    _y_test = _y_test['y_hat'].values

    # KFold
    cv_method = KFold(len(_y_train), number_KFold)

    grid = GridSearchCV(SVR(),
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
    #r_2 = best_model.scores(pred_train, _y_train)

    # error of test data
    # ------------------
    # --- 1 order ---
    bias = Error.bias(pred_test, _y_test)
    mae  = Error.mae(pred_test, _y_test)
    mse  = Error.mse(pred_test, _y_test)

    # --- 2 order ---
    rmse = Error.rmse(pred_test, _y_test)
    sde  = Error.sde(pred_test, _y_test)

    df_errors = pd.DataFrame({
                              #'r_2'  : [r_2],
                              'bias' : [bias],
                              'mae'  : [mae],
                              'mse'  : [mse],
                              'rmse' : [rmse],
                              'sde'  : [sde]
                             }, index=['h1_{0}'.format(horizon)])

    return pred_test, pred_forecast, df_errors










def svm2(_X_train, _Y_train, _X_test, _Y_test, kernels = ['rbf','linear', 'poly'],
        C=1e3, gamma=0.1):
    """

    :param _X:
    :param _y:
    :param _X_pred:
    :param _Y_pred:
    :param kernels:
    :param C:
    :param gamma:
    :return:
    """
    if len(_X_train.shape) is not 2: _X_train = _X_train[:, np.newaxis]
    if len(_X_test.shape) is not 2: _X_test = _X_test[:, np.newaxis]

    predL = []
    scores = []
    for i, kernel in enumerate(kernels):

        t1 = time.time()
        sv = SVR(kernel=kernel, C=C, gamma=gamma)
        # fit and predict
        pred = sv.fit(_X_train, _Y_train).predict(_X_test)
        t2 = time.time()
        CPU_time = t2 - t1


        predL.append(pred)
        # --- score
        scores.append('Bias {0} {1}'.format(kernel, Error.bias(pred, _Y_test)))
        scores.append('Mae {0} {1}'.format(kernel, Error.mae(pred, _Y_test)))
        #scores.append('MSE {0} {1}'.format(kernel, Error.mse(pred, _Y_test)))
        #scores.append('RMSE {0} {1}'.format(kernel, Error.rmse(pred, _Y_test)))
        scores.append('SMAPE {0} {1}'.format(kernel, Error.smape(pred, _Y_test)))


        if len(pred.shape) is not 2: pred = pred[:, np.newaxis]
        #scores.append('R2 {0} {1}'.format(kernel, sv.score(pred, _Y_test)))

        scores.append('CPU {0}'.format(CPU_time))
    return pred, scores



