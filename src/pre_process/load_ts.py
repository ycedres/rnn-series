
import pandas as pd

def load_ts(ts, date_parse=None, skiprows=None, delim_whitespace=None, header=None, column_names=None, parse_dates=None,
            index_col=None):

    if date_parse is not None:
        parse = lambda x: pd.datetime.strptime(x, date_parse)

    df = pd.read_csv(ts, skiprows=skiprows, delim_whitespace=delim_whitespace, header=header, names=column_names,
                     parse_dates=parse_dates, date_parser=parse, index_col=index_col)
    return df