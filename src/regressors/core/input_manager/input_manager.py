import pandas as pd
import numpy as np
import math
import datetime
from sklearn.model_selection import train_test_split

class InputManager(object):

    def InputManager(self):
        self._df = None
        self._train_set = None
        self._test_set = None
        self._validation_set = None

    def resample_ts(self,df, interval='H', operation='max'):
        df = df.resample(interval)

        if operation == 'max': return df.max()
        if operation == 'mean': return df.mean()


    def get_features(self,ts,date,**kwargs):
        print(type(kwargs))
        if 'method' in kwargs.keys():
            method = kwargs['method']
            del kwargs['method']

        df = pd.DataFrame()

        if method == 'sequential':
            df = self.get_features_sequential(ts,date,**kwargs)
        if method == 'daily':
            df = self.get_features_daily(ts,date,**kwargs)



        if 'write_csv_file' in kwargs.keys():
            write_csv_file = kwargs['write_csv_file']
        else:
            write_csv_file = False
        if 'output_csv_file' in kwargs.keys():
            output_csv_file = kwargs['output_csv_file']
        else:
            output_csv_file = None

        if write_csv_file is True:
            df.to_csv(output_csv_file, sep=';', float_format='%.2f', index=True)
            # df.to_csv(output_csv_file,sep=';')

        self._df = df
        self.dataframe_split()

        return df

    def get_features_sequential(self,ts, date=None, window_size=10, horizon=1, padding=0,
                     filename=None,step_size=1,write_csv_file=False,
                     output_csv_file=None):
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
            #ts = ts.ix[date]
            ts = ts.loc[date]

        upper_bound = len(ts) - (window_size + horizon + padding - 1)
        lower_bound = padding
        steps = math.ceil((upper_bound-lower_bound)/step_size)
        print("#################################################")
        print("SEQUENTIAL")
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
            if isinstance(ts.index[t],int):
                df_index[i]=ts.index[i]
            else:
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
        df.index.name = 'date'

        return df


    def get_features_daily(self,ts, date=None, window_size=10, horizon=1, padding=0,
                     filename=None,step_size=1,write_csv_file=False,
                     output_csv_file=None):
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
        lower_bound = window_size*24 + padding
        steps = math.ceil((upper_bound-lower_bound)/step_size)
        print("#################################################")
        print("DAILY")
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
        delta_horizon = pd.Timedelta('{} hours'.format(horizon))
        delta_ws_days= pd.Timedelta('1 days'*(window_size-1))
        for t in range(lower_bound, upper_bound,step_size):
            df_index[i]=ts.index[t].to_pydatetime()
            daily_window_start = ts.index[t]-delta_ws_days
            daily_window_indexes = pd.date_range(daily_window_start,
                                                 periods=window_size,freq='D')

            features[i][0: window_size] = ts[daily_window_indexes]
            h_offset = ts.index[t] + delta_horizon

            labels[i] = ts[h_offset]
            i += 1

        df_features = pd.DataFrame(features,
                            columns=['f_{0}'.format(i) for i in range(window_size)])

        df_target = pd.DataFrame({'target_h{0}'.format(horizon): labels})

        df = pd.concat([df_features, df_target], axis=1)

        df.index = df_index
        df.index.name = 'date'

        return df

    ####################################################################
    #DATAFRAME SPLIT

    def dataframe_split(self):
        """
        Split a pandas dataframe in three dataframe.
        :df: pandas dataframe
        :return: a dictionary where the value of the key is a pandas dataframe.
        each dataframe has a time measure, features and the target variable.
        """
        # split the original dataframe in two dataframe.
        # 80 % train and 20 % test set.
        train_size = int((self._df.shape[0] * 80) / 100)
        # Test set.
        df_train_set = self._df.iloc[0:train_size]
        df_test_set = self._df.iloc[train_size:]
        # Split train set in train and validation set using the method of
        # scikit-learn 'train_test_split'.
        df_train_set, df_validation_set = train_test_split(df_train_set,
                                                           test_size=0.2)

        self._train_set = df_train_set.fillna(0.0)
        self._test_set = df_test_set.fillna(0.0)
        self._validation_set = df_validation_set.fillna(0.0)

    def get_train_features(self):
        # features and target columns
        features_columns = self._train_set.columns[0:-1]
        # select the values of features and target columns.
        features_train_set = self._train_set[features_columns]
        return features_train_set

    def get_train_target(self):
        target_column = self._train_set.columns[-1:]
        target_train_set = self._train_set[target_column]
        return target_train_set

    def get_validation_features(self):
        features_columns = self._train_set.columns[0:-1]
        features_validation_set = self._validation_set[features_columns]
        return features_validation_set

    def get_validation_target(self):
        target_column = self._train_set.columns[-1:]
        target_validation_set = self._validation_set[target_column]
        return target_validation_set

    def get_test_features(self):
        features_columns = self._train_set.columns[0:-1]
        features_test_set = self._test_set[features_columns]
        return features_test_set

    def get_test_target(self):
        target_column = self._train_set.columns[-1:]
        target_test_set = self._test_set[target_column]
        return target_test_set


    def load_dataframe(self,data):
        self._df = data
        self.dataframe_split()

    def load_train_set(self,data):
        self._train_set = data

    def load_validation_set(self,data):
        self._validation_set = data

    def load_test_set(self,data):
        self._test_set = data

    def load_train_features(self,data):
        pass
    def load_train_target(self,data):
        pass
    def load_validation_features(self,data):
        pass
    def load_valication_target(self,data):
        pass
    def set_test_features(self,data):
        pass
    def set_test_target(self,data):
        pass
