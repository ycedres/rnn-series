import pandas as pd
import numpy as np
import math
import datetime

class InputManager(object):

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
            # df.to_csv('{0}_h_{1:0=2d}.csv'.format(filename, horizon),
            #           sep=';', float_format='%.2f', index=True)
            df.to_csv(output_csv_file,sep=';')

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
            ts = ts.ix[date]

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
