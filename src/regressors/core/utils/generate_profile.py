import pandas as pd
import pandas_profiling
df = pd.read_csv("/tmp/Offshore_WA_OR_features_h_12.csv",delimiter=";")
profile = pandas_profiling.ProfileReport(df)
profile.to_file(outputfile="/tmp/myoutputfile.html")

