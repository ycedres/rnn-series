"""
Data Engineering Pipeline

1.- Load file.
2.- Data clean.
3.- Resample the time serie.
4.- Generation of features for ML and ANN algorithms.
"""
from src.pre_process.load_ts import load_ts
from src.pre_process.resample_ts import resample_ts
from src.pre_process.get_features import get_features

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