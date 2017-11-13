__author__ = 'tmorales'

"""
Forecast Strategies
--------------------

A.- ) One-ahead and iterated prediction.

B.-) Multi-step a head prediction.

    B.1.-) Iterared

        B.1.1.-) Parameters are optimized to minimizer training error on one-step-ahead forecast.

        B.1.2.-) Parameters are optimized to minimizer training error on the iterated h-step-ahead forecast.

    B.2.-) Direct

    B.3.-) DiRec

    B.4.-) MIMO


"""


import pandas as pd

from preProceso.powerMapping import features_generator
from preProceso.powerMapping import labels_generator


#
# A.-) ONE-AHEAD AND ITERATED PREDICTION
#



#
# B.-) MULTI-STEP AHEAD PREDICTION
#
def direct_strategy(ts, feature_window=20, horizon=1):
    """
    :param ts: serie temporal original
    :param feature_window: tamano de la ventana
    :param horizon: horizonte
    :return:
    """
    #
    # time index of original time serie
    #
    timerange = ts.index

    # Feature window [examples, feature_window]
    features = features_generator(ts,
                                  feature_window=feature_window,
                                  horizont=horizon)
    # Label [examples,]
    labels = labels_generator(ts,
                              feature_window=feature_window,
                              horizont=horizon)

    # DataFrame [examples - feature_window, label]
    y_hat = pd.DataFrame(labels,
                         columns=['y_hat'],
                         index = timerange[feature_window+horizon-1:])

    return features, y_hat


def diRec_strategy():
    pass

def mIMO_strategy():
    pass