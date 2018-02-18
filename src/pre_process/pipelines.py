"""
Data Engineering Pipeline

1.- Load file.
2.- Data clean.
3.- Resample the time serie.
4.- Generation of features for ML and ANN algorithms.
"""
import os
import pandas as pd

from load_ts import load_ts
from resample_ts import resample_ts
from get_features import get_features


def pozo_izquierdo_ts():
    # -- 1.- Load file
    torre_me = '../../database/torrePozoIzquierdo_JUL_AGO_2004/torreME.txt'

    df = load_ts(torre_me, date_parse='%d/%m/%Y %H:%M:%S', skiprows=12, delim_whitespace=True, header=None,
                column_names=['day', 'time', 'direccion_viento', 'v_h10', 'v_h20', 'v_h40'],
                parse_dates=[['day', 'time']],
                index_col=0)

    df_40 = df[['v_h40']]

    # -- 2.- Data clean

    # -- 3.- Resample the ts
    df_40_H = resample_ts(df_40, operation='mean')

    # -- 4.- Generation of features for ML and ANN algorithms.
    ts = df_40_H['v_h40']
    date = df_40_H.index

    ml_filename = '../../database/torrePozoIzquierdo_JUL_AGO_2004/torreME_features/pozo_izquierdo_torre_me'


    for h in range(1, 25):
        df_40_H_h1 = get_features(ts, date, window_size=10, horizont=h,
                            write_csv_file=True,
                            filename=ml_filename)
        print('Filename with h = {0} created ...'.format(h))


def wind_power_parks(filename,**kwargs):
    
    if 'step_size' in kwargs.keys():
        step_size = kwargs['step_size']
    if 'padding' in kwargs.keys():
        padding = kwargs['padding']
    if 'horizon' in kwargs.keys():
        horizon = kwargs['horizon']
    if 'window_size' in kwargs.keys():
        window_size = kwargs['window_size']

    # 1.- Load file
    parse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    turbine_25915 = pd.read_csv(filename,
                                 sep=';',
                                 parse_dates=['Date(YYYY-MM-DD hh:mm:ss)'],
                                 date_parser=parse,
                                 index_col=0)
    turbine_25915.index.names = ['date']
    turbine_25915 = turbine_25915.rename(columns={'100m wind speed (m/s)': 'wind_speed_100m_m/s',
                                                   ' rated power output at 100m (MW)': 'rated_power_output_100m_MW',
                                                   ' SCORE-lite power output at 100m (MW)': 'SCORE-lite_power_100m_MW',
                                                   'CorrectedScore': 'Corrected_Score'})

    # 3.- Resample the ts
    turbine_25915_wind = turbine_25915[['wind_speed_100m_m/s']]
    turbine_25915_wind_H = resample_ts(turbine_25915_wind, interval='H', operation='mean')

    # -- 4.- Generation of features for ML and ANN algorithms.
    ts = turbine_25915_wind_H['wind_speed_100m_m/s']
    date = turbine_25915_wind_H.index

    ml_filename = '/tmp/Offshore_WA_OR_features'

    for h in range(12, 13):
        get_features(ts, date, window_size=window_size, horizon=h,
                     write_csv_file=True,
                     filename=ml_filename,
                     padding=padding,
                     step_size=step_size)
        print ('Filename with h = {0} created ...'.format (h))



if __name__ == "__main__":
    #pozo_izquierdo_ts()
    filename = '/home/ycedres/Projects/PhD/wind/RNN-windPower/database/windpark_Offshore_WA_OR_turbine_25915.csv'
    wind_power_parks(filename=filename)
