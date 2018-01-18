
from sklearn.cross_validation import train_test_split

def dataframe_split(df):
    """
    Split a pandas dataframe in three dataframe.
    :df: pandas dataframe
    :return: a dictionary where the value of the key is a pandas dataframe.
             each dataframe has a time measure, features and the target variable.
    """
    # split the original dataframe in two dataframe.
    # 80 % train and 20 % test set.
    train_size = int((df.shape[0] * 80) / 100)
    # Test set.
    df_train_set = df.iloc[0:train_size]
    df_test_set = df.iloc[train_size:]
    # Split train set in train and validation set using the method of
    # scikit-learn 'train_test_split'.
    df_train_set, df_validation_set = train_test_split(df_train_set, test_size=0.2)

    return {'train_set': df_train_set,
            'validation_set': df_validation_set,
            'test_set': df_test_set}