import json
import glob
import pandas as pd

res = {}
df = pd.DataFrame()

for filename in glob.iglob('/tmp/**/*.json', recursive=True):
    print(filename)
    with open(filename) as json_data:
        d = json.load(json_data)
        df.from_dict(d)
        print(df)


        #res = {**res,**d}
        #res = {key: value for (key, value) in (res.items() + d.items())}
