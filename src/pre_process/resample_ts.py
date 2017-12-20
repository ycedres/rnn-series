import pandas as pd


def resample_ts(df, interval='H', operation='max'):
    df = df.resample(interval)

    if operation == 'max': return df.max()
    if operation == 'mean': return df.mean()
