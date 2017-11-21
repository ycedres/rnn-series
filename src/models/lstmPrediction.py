
#
#
# MAL ----> no ejecuta desde los IPython notebooks
#
#



from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor


class LSTM(object):

    #def __init__(self, featuresByTimestep=20):
    #    self.input_dim = featuresByTimestep

    def create_topology(self, optimizer='adam', init='uniform', nh=4, inputByTimeStep=2):
        # --- create model
        model = Sequential()
        model.add(LSTM(nh, input_dim=inputByTimeStep))
        model.add(Dense(1))

        # --- compile model
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def lstm_fit(self, splitDataset, paramGrid):

        # --- Model
        model = KerasRegressor(build_fn=self.create_topology)

        # --- Grid Search
        grid = GridSearchCV(estimator=model,
                            param_grid=paramGrid
                        )

        # --- Fit the model
        grid_result = grid.fit(splitDataset['xTrain'], splitDataset['yTrain'])

        print('Aqui ----')

        # --- Best model
        best_model = grid_result.best_estimator_
        print('Best model: {0}\nBest score: {1} '.format(grid_result.best_params_,
                                                     grid_result.best_score_))
        scoreAllModels = grid_result.grid_scores_

        # --- Predicted model
        trainPredicted = best_model.predict(splitDataset['xTrain'])
        testPredicted = best_model.predict(splitDataset['xTest'])

        predicted = {'trainPredicted' : trainPredicted, 'testPredicted' : testPredicted}

        return predicted, scoreAllModels
