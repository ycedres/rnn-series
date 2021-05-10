"""
Data Engineering Pipeline

1.- Load file.
2.- Data clean.
3.- Resample the time serie.
4.- Generation of features for ML and ANN algorithms.
"""
import os
import pandas as pd

from input_manager.input_manager import InputManager

class SupplyChainInputManager(InputManager):

    def SupplyChainInputManager(self,method=None,**kwargs):
        self._ts = None
        self._ts_ref = None
        if 'periods_reference' in kwargs.keys():
            self._periods_reference = kwargs['periods_reference']

        else:
            self._periods_reference = None
        pass

    def configure_load_datasource(self,method,colindex=0,**kwargs):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"+ str(colindex))

        self._colindex = colindex        
        print("#######################" + str(self._colindex))        
        if method=='filesystem':
            if 'filename' in kwargs.keys():
                self._ts = self.get_ts(filename=kwargs['filename'])
                self._ts_ref = self._ts.shift(periods=1)




    def configure_features_generator(self,**kwargs):

        if 'window_size' in kwargs.keys():
            self._window_size = kwargs['window_size']
        if 'horizon' in kwargs.keys():
            self._horizon = kwargs['horizon']
        if 'padding' in kwargs.keys():
            self._padding = kwargs['padding']
        if 'step_size' in kwargs.keys():
            self._step_size = kwargs['step_size']

        if 'write_csv_file' in kwargs.keys():
            self._write_csv_file = kwargs['write_csv_file']
        else:
            write_csv_file = False
        if 'output_csv_file' in kwargs.keys():
            self._output_csv_file = kwargs['output_csv_file']
        else:
            self._output_csv_file = None

        if 'method' in kwargs.keys():
            self._method = kwargs['method']


    def read_input_features(self):
        return self._input_manager.load_data()

    def read_target(self):
        return self._input_manager.read_target()

    def get_ts(self,filename):
        df = pd.read_csv(filename,header=None)
        ts = df.iloc[:,self._colindex]
        return ts


    def load_and_split(self):


        #ts = self.get_ts(filename)

        # print("######################## TIME SERIE ORIGINAL")
        # print(type(ts))
        # print(ts[0:50])
        # print("########################")
        date = self._ts.index

        # for h in range(12, 13):
        #     get_features(ts, date, window_size=window_size, horizon=h,
        #                  write_csv_file=True,
        #                  filename=ml_filename,
        #                  padding=padding,
        #                  step_size=step_size)

        self._df = self.get_features(self._ts, date,
                     window_size=self._window_size,
                     horizon=self._horizon,
                     padding=self._padding,
                     step_size=self._step_size,
                     write_csv_file=self._write_csv_file,
                     output_csv_file=self._output_csv_file,
                     method=self._method)


    def get_features_target(self):
        return self._df


    def get_reference_serie(self,type='persistence',periods=None):
        if type=='persistence':
            if periods is not None:
                self._ts_ref = self._ts.shift(periods=periods)
            elif self._periods_reference is not None:
                self._ts_ref = self._ts.shift(periods=self._periods_reference)
            else:
                self._ts_ref = self._ts.shift(periods=1)

        return self._ts_ref

    def get_reference_rmse(self):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(self._ts[1:],self._ts_ref[1:])

if __name__ == "__main__":
    #pozo_izquierdo_ts()
    base_dir = '/home/ycedres/Projects/RNN/RNN-windPower/database/'
    filename = 'windpark_Offshore_WA_OR_turbine_25915.csv'

    im = NRELInputManager()

    im.configure_load_datasource(method='filesystem',filename=base_dir+filename)

    im.configure_features_generator(
        window_size=10,
        horizon=12,
        padding=0,
        step_size=1,
        write_csv_file=True,
        output_csv_file='/tmp/output_csv_file.csv',
        #method='sequential',
        method='daily'
    )

    im.load_and_split()

    print(im.get_test_features())
