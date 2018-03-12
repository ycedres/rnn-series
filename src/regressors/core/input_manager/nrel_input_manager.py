"""
Data Engineering Pipeline

1.- Load file.
2.- Data clean.
3.- Resample the time serie.
4.- Generation of features for ML and ANN algorithms.
"""
import os
import pandas as pd

from input_manager import InputManager

class NRELInputManager(InputManager):

    def NRELInputManager(self,method=None,**kwargs):
        #TODO Parámentros: parque, turbina, rango temporal...
        #Fuente: parámetros de conexión, ruta fichero...
        pass

    def read_input_features(self):
        return self._input_manager.load_data()

    def read_target(self):
        return self._input_manager.read_target()

    def get_ts(self,filename):

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
        turbine_25915_wind_H = self.resample_ts(turbine_25915_wind, interval='H', operation='mean')

        # -- 4.- Generation of features for ML and ANN algorithms.
        ts = turbine_25915_wind_H['wind_speed_100m_m/s']

        return ts


    def get_features_target(self,filename,**kwargs):

        # Parameters for get_features
        if 'window_size' in kwargs.keys():
            window_size = kwargs['window_size']
        if 'horizon' in kwargs.keys():
            horizon = kwargs['horizon']
        if 'padding' in kwargs.keys():
            padding = kwargs['padding']
        if 'step_size' in kwargs.keys():
            step_size = kwargs['step_size']

        if 'write_csv_file' in kwargs.keys():
            write_csv_file = kwargs['write_csv_file']
        else:
            write_csv_file = False
        if 'output_csv_file' in kwargs.keys():
            output_csv_file = kwargs['output_csv_file']
        else:
            output_csv_file = None

        if 'method' in kwargs.keys():
            method = kwargs['method']


        ts = self.get_ts(filename)

        # print("######################## TIME SERIE ORIGINAL")
        # print(type(ts))
        # print(ts[0:50])
        # print("########################")
        date = ts.index

        # for h in range(12, 13):
        #     get_features(ts, date, window_size=window_size, horizon=h,
        #                  write_csv_file=True,
        #                  filename=ml_filename,
        #                  padding=padding,
        #                  step_size=step_size)

        df = self.get_features(ts, date, window_size=window_size, horizon=horizon,
                     padding=padding,
                     step_size=step_size,
                     write_csv_file=write_csv_file,
                     output_csv_file=output_csv_file,
                     method=method)

        return df


if __name__ == "__main__":
    #pozo_izquierdo_ts()
    base_dir = '/home/ycedres/Projects/RNN/RNN-windPower/database/'
    filename = 'windpark_Offshore_WA_OR_turbine_25915.csv'

    im = NRELInputManager()

    df = im.get_features_target(filename=base_dir+filename,window_size=10,
                               horizon=12,
                               padding=0,step_size=1,
                               write_csv_file=True,
                               output_csv_file='/tmp/output_csv_file.csv',
                               #method='sequential')
                               method='daily')
