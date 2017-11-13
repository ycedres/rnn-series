__author__ = 'tmorales'

import numpy as np
import pandas as pd

def features_generator(ts, feature_window=10, horizont=1, padding=0):
    '''
    ts : Pandas DataFrame
    '''
    timesteps = len(ts) - (feature_window + horizont + padding - 1)

    features = np.zeros((timesteps, feature_window), dtype = np.float32)

    for t in range(padding, timesteps):
        # features
        features[t][0: feature_window] = ts[t : t+feature_window]

    return features


def labels_generator(ts, feature_window=10, horizont=1, padding=0):
    '''
    ts : Pandas DataFrame
    '''
    timesteps = len(ts) - (feature_window + horizont + padding - 1)

    labels = np.zeros((timesteps), dtype=np.float32)

    for t in range(padding, timesteps):
        # labels
        offset = t + feature_window + horizont - 1
        labels[t] = ts[offset]

    return labels


# quitar y poner en preProceso.windowMethod ---------------------------

def window_method(ts, feature_window=20, horizon=1):
    #
    # time index of original time serie
    #
    #timerange = ts.index

    # Feature window [examples, feature_window]
    features = features_generator(ts,
                                  feature_window=feature_window,
                                  horizont=horizon)
    # Label [examples,]
    labels = labels_generator(ts,
                              feature_window=feature_window,
                              horizont=horizon)

    # DataFrame [examples - feature_window, label]
    #y_hat = pd.DataFrame(labels,
    #                     columns=['y_hat'],
    #                     index = timerange[feature_window+horizon-1:])

    return features, labels
           #y_hat
