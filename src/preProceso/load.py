__author__ = 'tmorales'

"""
Carga diferentes feuntes de datos
"""

import pandas as pd

def wind_power_capacity(fileName):
    try:
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        ts = pd.read_csv(fileName, delimiter=';',
                         parse_dates=['GMT'],
                         date_parser=dateparse,
                         index_col=0,
                         dtype={'windCapacity': 'float32'}
                        )
    except:
        print('The fail has not found!!!')

    return ts


def pv_power_capacity(fileName):
    try:
        pass
    except:
        pass