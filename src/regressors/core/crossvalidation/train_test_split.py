
class TrainTestSplit(object):

    def __init__(self, df):
        self._df = df
        self._train_set = None
        self._test_set = None
        self._validation_set = None

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
        df_train_set = df.iloc[0:train_size]
        df_test_set = df.iloc[train_size:]
        # Split train set in train and validation set using the method of
        # scikit-learn 'train_test_split'.
        df_train_set, df_validation_set = train_test_split(df_train_set,
                                                           test_size=0.2)

        df._train_set = df_train_set
        df._test_set = df_test_set
        df._validation_set = df_validation_set

    def csv_split(self):
        pass

    def get_features(self):
        # features and target columns
        features_columns = df._train_set.columns[1:-1]
        # select the values of features and target columns.
        features_train_set = df._train_set[features_columns]
        return features_train_set

    def get_target(self):
        target_column = df._train_set.columns[-1:]
        target_train_set = df._train_set[target_column]
        return target_train_set

    def get_validation(self):
        features_validation_set = df._validation_set[features_columns]
        target_validation_set = df._validation_set[target_column]
        return target_validation_set
