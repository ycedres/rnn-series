"""
Data Engineering Pipeline

1.- Load file.
2.- Data clean.
3.- Resample the time serie.
4.- Generation of features for ML and ANN algorithms.
"""
import os
import pandas as pd

from src.pre_process.load_ts import load_ts
from src.pre_process.resample_ts import resample_ts
from src.pre_process.get_features import get_features


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


def wind_power_parks():
    filename = '/Users/tmorales/tmp/RNN-windPower/database/wind_farms/Offshore_WA_OR/windpark_Offshore_WA_OR_turbine_25915.csv'

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

    ml_filename = '/Users/tmorales/tmp/RNN-windPower/database/wind_farms/Offshore_WA_OR/Offshore_WA_OR_features'

    for h in range(1, 25):
        get_features(ts, date, window_size=6, horizont=h,
                     write_csv_file=True,
                     filename=ml_filename)
        print ('Filename with h = {0} created ...'.format (h))



if __name__ == "__main__":
    #pozo_izquierdo_ts()
    wind_power_parks()