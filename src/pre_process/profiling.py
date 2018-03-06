import pipelines
import pandas as pd
import pandas_profiling
from split_ml_ts import dataframe_split

base_dir = '/home/ycedres/Projects/RNN/RNN-windPower/database/'
input_filename = 'windpark_Offshore_WA_OR_turbine_25915.csv'

df = pipelines.wind_power_parks(filename=base_dir+input_filename,
                           window_size=10,horizon=12,
                           padding=0,step_size=10,
                           write_csv_file=True,
                           output_csv_file='/tmp/Offshore_WA_OR_features_h_12.csv',
                           method='sequential')
                           #method='daily')

# df = pd.read_csv("/tmp/Offshore_WA_OR_features_h_12.csv",
#                 sep=';',
#                 header=0,
#                 index_col=0
#                 )

data = dataframe_split(df)

train_set = data['train_set']
validation_set = data['validation_set']
test_set = data['test_set']
#
# #train_set.to_csv('/tmp/train_set.csv',sep=';')
profile = pandas_profiling.ProfileReport(train_set)
profile.to_file(outputfile="/tmp/myoutputfile.html")
