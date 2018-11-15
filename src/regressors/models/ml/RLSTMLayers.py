import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Add, Lambda
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from .AbstractModel import AbstractModel
from keras.layers import Conv1D,GlobalAveragePooling1D, MaxPooling1D

import os

class RLSTMLayers(AbstractModel):

    def __init__(self, config=None,basedir=None,file_prefix=None,
                 input_descriptor_string=None):

        if basedir is not None:
            self._basedir = basedir
        if file_prefix is not None:
            self._file_prefix = file_prefix
        if input_descriptor_string is not None:
            self._input_descriptor_string = input_descriptor_string

        self._plot_title = 'lstm'
        #self._reg = self._create_model(10, 1)
        #self._batch_size = 1024
        #self._epochs = 100

    def config_exp_path(self,basedir=None,file_prefix=None,
                 input_descriptor_string=None):

        self._basedir = basedir
        self._file_prefix = file_prefix
        self._input_descriptor_string = input_descriptor_string


    def configure(self,timesteps):
        self._reg = self._create_model(timesteps , 1)

    def configure_train(self,config):
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']

    def set_batch_size(self,batch_size):
        self._batch_size = batch_size

    def set_epochs(self,epochs):
        self._epochs = epochs

    def train(self, features_train_set,
                    target_train_set,
                    features_validation_set,
                    target_validation_set):
        print("TRAAAAAAAAAAAAAAAAAAAAAAAIN")
        x_train = features_train_set.values.astype('float32')
        y_train = target_train_set.values
        x_val = features_validation_set.values.astype('float32')
        y_val = target_validation_set.values

        x_train_lstm = np.reshape(x_train, (x_train.shape[0],
                                            x_train.shape[1], 1))

        x_val_lstm = np.reshape(x_val, (x_val.shape[0],
                                        x_val.shape[1], 1))

        directory = self._basedir + '/' + self._file_prefix + '_' + \
        self._input_descriptor_string + '/'

        model, history = self._fit_model(self._reg, x_train_lstm, y_train,
                                          x_val_lstm, y_val,
                                          exp_path=directory,
                                          batch_size=self._batch_size,
                                          epochs=self._epochs)

    def test(self,features_test_set):

        x_test = features_test_set.values.astype('float32')
        x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # print("TEEEEEEEEEEEEEEEEEEEEEST")
        # print(self._reg.layers)
        # import sys
        # sys.exit()
        return self._reg.predict(x_test_lstm)



    def _create_model(self,time_step, feature_by_timestep):
        # Input layer
        input = Input((time_step,
                       feature_by_timestep),
                       dtype='float32',
                       name='input-layer')
        #normalize_input = BatchNormalization(name='normalize-input')(input)
        x = Conv1D(50, 1, activation='relu', input_shape=(time_step, feature_by_timestep))(input)
        x = Conv1D(50, 1, activation='relu', input_shape=(time_step, feature_by_timestep))(x)
        #Timesteps x número de filtros (timesteps x features) entran a la LSTM
        print("@@@@@@@@@@@@@@@@@@ÚLTIMA CONV1D: ",x)
        x = LSTM(90,
                 kernel_initializer='normal',
                 activation='relu',
                 name='lstm-layer')(x)

        print("@@@@@@@@@@@@@@@@@@LSTM: ", x)

        #x = Dropout(rate=0.2)(x)
        x = Dense(50,
                       kernel_initializer='normal',
                       activation='relu',
                       name='output-layer-1')(x)
        print("@@@@@@@@@@@@@@@@@@DENSA 1: ", x)
        x = Dense(50,
                       kernel_initializer='normal',
                       activation='relu',
                       name='output-layer-2')(x)
        print("@@@@@@@@@@@@@@@@@@DENSA 2: ", x)
        x = Dense(1,
                       kernel_initializer='normal',
                       activation='linear',
                       name='output-layer-3')(x)
        print("@@@@@@@@@@@@@@@@@@DENSA 3: ", x)
        xt = Lambda(lambda x: x[:,-1,:], output_shape=(1,))(input)
        print("@@@@@@@@@@@@@@@@@@XT: ", x)
        output = Add()([x,xt])

        # Model
        model = Model(inputs=input, outputs=output)
        model.summary()

        plot_model(model,
                   to_file='{0}.png'.format(self._plot_title),
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='LR')

        return model


    def _fit_model(self,model, x_train, y_train, x_val, y_val, exp_path,
                  epochs=100, batch_size=50, lr=0.001):

        tensorboard = TensorBoard(log_dir=os.path.join(exp_path, 'graph'),
                               histogram_freq=1)

        opt = Adam(lr=lr)
        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['mae'])

        history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val)
                            #callbacks=[tensorboard]
                            )

        return model, history


    def plot_model(self):

        directory = self._basedir + '/' + self._file_prefix + '_' + \
        self._input_descriptor_string + '/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        plot_model(self._reg,
                   to_file=directory+'{0}.png'.format(self._plot_title),
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='LR')


if __name__ == '__main__':
    r = RLSTM()
    r._create_model(6, 1, plot_title='lstm')
