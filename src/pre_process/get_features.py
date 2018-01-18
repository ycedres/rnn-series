
import numpy as np
import pandas as pd

def get_features(ts, date=None, window_size=10, horizont=1, padding=0,
                 write_csv_file=False, filename=None):
    """

    :param ts:
    :param date:
    :param window_size:
    :param horizont:
    :param padding:
    :param write_csv_file:
    :param filename:
    :return:
    """
    # target index (date)
    if date is not None:
        date = date[window_size + horizont - 1:]
        df_target_date = pd.DataFrame({'target_date': date})
    # features
    timesteps = len(ts) - (window_size + horizont + padding - 1)
    features = np.zeros((timesteps, window_size), dtype=np.float32)
    for t in range(padding, timesteps):
        features[t][0: window_size] = ts[t: t + window_size]
    df_features = pd.DataFrame(features,
                               columns=['f_{0}'.format(i) for i in range(window_size)])
    # labels
    labels = np.zeros((timesteps), dtype=np.float32)
    for t in range(padding, timesteps):
        offset = t + window_size + horizont - 1
        labels[t] = ts[offset]
    df_target = pd.DataFrame({'target_h{0}'.format(horizont): labels})

    # concat all df
    df = pd.concat([df_target_date, df_features, df_target], axis=1)

    if write_csv_file is True:
        df.to_csv('{0}_h_{1:0=2d}.csv'.format(filename, horizont),
                  sep=';', float_format='%.2f', index=False)
    return df