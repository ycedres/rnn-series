
import numpy as np

from sklearn import neighbors

from utils import Error

def KNeighbors(_X_train, _y_train, _X_test, _Y_test, n_neighbors=5,
               weights=['uniform', 'distance']):
    """
    :return:
    """
    if len(_X_train.shape) is not 2: _X_train = _X_train[:, np.newaxis]
    if len(_X_test.shape) is not 2: _X_test = _X_test[:, np.newaxis]
    print(_X_train.shape)

    predL = []
    scores = []
    for i, weight in enumerate(weights):
        kk = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights=weight)
        # fit and predict
        pred = kk.fit(_X_train, _y_train).predict(_X_test)

        #print(pred)

        predL.append(pred)
        # --- score
        scores.append('Bias {0} {1}'.format(weight, Error.bias(pred, _Y_test)))
        scores.append('Mae {0} {1}'.format(weight, Error.mae(pred, _Y_test)))
        #scores.append('MSE {0} {1}'.format(weight, Error.mse(pred, _Y_test)))
        #scores.append('RMSE {0} {1}'.format(weight, Error.rmse(pred, _Y_test)))
        scores.append('SMAPE {0} {1}'.format(weight, Error.smape(pred, _Y_test)))


        if len(pred.shape) is not 2: pred = pred[:, np.newaxis]
        #scores.append('R2 {0} {1}'.format(weight, kk.score(pred, _Y_test)))

    return predL, scores




