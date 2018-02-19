
import numpy as np
import pandas as pd
import datetime
import math

def get_features(ts, date=None, window_size=10, horizon=1, padding=0,
                 write_csv_file=False, filename=None,step_size=1):
    """

    :param ts:
    :param date:
    :param window_size:
    :param horizon:
    :param padding:
    :param write_csv_file:
    :param filename:
    :return:
    """

    # target index (date)
    # if date is not None:
    #     date = date[window_size + horizon - 1:]
    #     df_target_date = pd.DataFrame({'target_date': date})

    # TODO: Comprobar
    if date is not None:
        ts = ts.ix[date]

    upper_bound = len(ts) - (window_size + horizon + padding - 1)
    lower_bound = padding
    steps = math.ceil(upper_bound/step_size)
    print("#################################################")
    print("len(ts): {}".format(len(ts)))
    print("window size: {}".format(window_size))
    print("padding: {}".format(padding))
    print("horizon: {}".format(horizon))
    print("upper_bound: {}".format(upper_bound))
    print("lower_bound: {}".format(lower_bound))
    print("step_size: {}".format(step_size))
    print("steps: {}".format(steps))
    print("#################################################")

    features = np.zeros((steps, window_size), dtype=np.float32)
    labels = np.zeros(steps, dtype=np.float32)
    df_index = np.zeros(steps,dtype=datetime.datetime)

    i = 0
    for t in range(lower_bound, upper_bound,step_size):
        df_index[i]=ts.index[t].to_pydatetime()
        features[i][0: window_size] = ts[t: t + window_size]
        h_offset = t + window_size + horizon - 1
        labels[i] = ts[h_offset]
        i += 1

    df_features = pd.DataFrame(features,
                        columns=['f_{0}'.format(i) for i in range(window_size)])

    df_target = pd.DataFrame({'target_h{0}'.format(horizon): labels})

    df = pd.concat([df_features, df_target], axis=1)

    df.index = df_index

    if write_csv_file is True:
        df.to_csv('{0}_h_{1:0=2d}.csv'.format(filename, horizon),
                  sep=';', float_format='%.2f', index=True)

    print(df.shape)
    print(df.head())
    return df
