import pipelines
import pandas as pd
import pandas_profiling
from split_ml_ts import dataframe_split

filename = '/home/ycedres/Projects/PhD/wind/RNN-windPower/database/windpark_Offshore_WA_OR_turbine_25915.csv'
pipelines.wind_power_parks(filename=filename,window_size=10,horizon=12,
                           padding=0,step_size=10)

df = pd.read_csv("/tmp/Offshore_WA_OR_features_h_12.csv",delimiter=";")

data = dataframe_split(df)
train_set = data['train_set']
validation_set = data['validation_set']
test_set = data['test_set']

profile = pandas_profiling.ProfileReport(train_set)
profile.to_file(outputfile="/tmp/myoutputfile.html")
