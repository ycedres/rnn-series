

from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


class MLP():

    #def __init__(self, window_features=20):
    #    self.window_features = window_features

    def create_topology(self, optimizer='adam', init='uniform', window_features=25, nh=8):
        # --- create model
        model = Sequential()
        model.add(Dense(nh, input_dim=window_features, activation='relu'))
        model.add(Dense(1))

        # --- compile model
        model.compile(loss='mean_squared_error', optimizer='adam', verbose=1)

        return model


    def mlp_fit(self, splitDataset, paramGrid):

        print(paramGrid)

        # -- Model
        model = KerasRegressor(build_fn=self.create_topology, verbose=1)

        # -- Grid search
        grid = GridSearchCV(estimator=model, param_grid=paramGrid, verbose=1)

        # -- Fit the model
        grid_result = grid.fit(splitDataset['xTrain'], splitDataset['yTrain'])

        # -- Best model
        best_model  = grid_result.best_estimator_
        best_params = grid_result.best_params_
        print('Best model: {0}\nBest score: {1} '.format(grid_result.best_params_,
                                                     grid_result.best_score_))
        scoreAllModels = grid_result.grid_scores_

        # -- Predicted values
        trainPredicted = best_model.predict(splitDataset['xTrain'])
        testPredicted = best_model.predict(splitDataset['xTest'])


        predicted = {'trainPredicted' : trainPredicted, 'testPredicted' : testPredicted}


        return best_params, scoreAllModels

    def mlp_run(self, bestModel):
        # Best model
        paramGrid = {}
        for i in bestModel.iteritems():
            paramGrid[i[0]] = [i[1]]

        print(paramGrid)



