from sklearn.neighbors import KNeighborsRegressor

class KNNRegressor(object):

    def __init__(self, config):
        self._config = config
        # instance the class
        n_neighbors = int(self._config['n_neighbors'])
        weight = self._config['weight']

        self._knr = KNeighborsRegressor(n_neighbors=n_neighbors,weights=weight)

    def train(self,data,data_test):
        x_train = data[0]
        y_train = data[1]
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        # print(type(x_train))
        # print(x_train.shape)
        # #print(x_train)
        # print('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
        # print(type(y_train))
        # print(y_train.shape)

        ##x_train, y_train = data['training_data']
        #self._knr = KNeighborsRegressor(n_neighbors=5,weights='uniform')
        #print(self._knr.fit(x_train,y_train).predict(data_test[0]))
        self._knr.fit(x_train,y_train)

    def test(self,train_data,data):
        # print("PREDICTING")
        # print(data[0])
        # print(type(data[0]))
        # print(data[0].shape)
        x_train = train_data[0]
        y_train = train_data[1]
        self._knr = KNeighborsRegressor(n_neighbors=5,weights='uniform')
        self._knr.fit(x_train,y_train)
        x_test = data[0]
        #print(self._knr.predict(x_test))
        return self._knr.predict(x_test)
