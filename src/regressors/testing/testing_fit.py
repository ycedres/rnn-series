import pandas as pd

from sklearn.cross_validation import train_test_split


# **********************************************************
#
# pre - process
#
# **********************************************************
def load_data(filename):
    # load dataset in dataframe
    return pd.read_csv(filename, delimiter=';')


def split_dataset(df):
    # Split the dataset
    train_set, test_set = train_test_split(df, test_size=0.2)
    train_set, validation_set = train_test_split(train_set, test_size=0.2)

    return train_set, test_set, validation_set


def get_table_target(test_set, save_csv=False):
    # dataset predict -- inside the experiment folder
    target_table = test_set[['reg', 'target']]
    target_table = target_table.sort_values(by=['reg'])

    if save_csv == True:
        target_table.to_csv('target_table.csv', sep=';', index=False)

    return target_table


def get_data(filename):

    # load the dataset
    df = load_data (filename)
    # split the dataset
    train_set, test_set, val_set = split_dataset (df)
    # target table
    target_table = get_table_target(test_set, save_csv=True)

    # get data
    data = {}
    # feature columns
    features_columns = list(set(train_set.columns) - {'reg', 'taget'})
    # -- dataset for training
    training_data = []
    # train set - features
    train_features = train_set[features_columns].values
    training_data.insert(0, train_features)
    # train set - label
    train_label = train_set['target'].values
    training_data.insert(1, train_label)
    # load in data
    data['training_data'] = training_data

    # -- dataset for validation
    validation_data = []
    # validation set - features
    validation_features = val_set[features_columns].values
    validation_data.insert(0, validation_features)
    # validation set - label
    validation_label = val_set['target'].values
    validation_data.insert(1, validation_label)
    # load in data
    data['validation_data'] = validation_data

    # -- testing data
    testing_data = []
    # testing set - features
    test_set = test_set.sort_values(by=['reg'])
    testing_features = test_set[features_columns].values
    testing_data.insert(0, testing_features)
    data['testing_data'] = testing_data

    return data, target_table



if __name__ == "__main__":
    filename = '../data/boston_dataset.csv'
    df = load_data(filename)


    train_set, test_set, validation_set = split_dataset(df)
    #print(test_set)

    target_table = get_table_target(test_set, save_csv=True)

    data = get_data(train_set, test_set, validation_set)

    print(data)

