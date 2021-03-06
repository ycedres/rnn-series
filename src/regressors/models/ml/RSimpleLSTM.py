import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import os

class RSimpleLSTM(object):

    def __init__(self, config=None,basedir=None,file_prefix=None,
                 input_descriptor_string=None):
                 
        if basedir is not None:
            self._basedir = basedir
        if file_prefix is not None:
            self._file_prefix = file_prefix
        if input_descriptor_string is not None:
            self._input_descriptor_string = input_descriptor_string

        self._plot_title = 'simple_lstm'
        self._reg = self._create_model(6, 1)

    def config_exp_path(self,basedir=None,file_prefix=None,
                 input_descriptor_string=None):

        self._basedir = basedir
        self._file_prefix = file_prefix
        self._input_descriptor_string = input_descriptor_string

    def train(self, features_train_set,
                    target_train_set,
                    features_validation_set,
                    target_validation_set):

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
                                          batch_size=1024, epochs=100)

    def test(self,features_test_set):

        x_test = features_test_set.values.astype('float32')
        x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return self._reg.predict(x_test_lstm)



    def _create_model(self,time_step, feature_by_timestep):
        # Input layer
        input = Input((time_step,
                       feature_by_timestep),
                       dtype='float32',
                       name='input-layer')
        #normalize_input = BatchNormalization(name='normalize-input')(input)

        # RNN - LSTM
        x = LSTM(6,
                 kernel_initializer='normal',
                 name='lstm-layer')(input)
        # Fully-connect
        # x = Dense(10,
        #           kernel_initializer='normal',
        #           activation='relu',
        #           name='hidden-layer-1')(x)
        # x = Dense(5,
        #           kernel_initializer='normal',
        #           activation='relu',
        #           name='hidden-layer-2')(x)
        # x = Dense(4,
        #           kernel_initializer='normal',
        #           activation='relu',
        #           name='hidden-layer-3')(x)
        # x = Dense(3,
        #           kernel_initializer='normal',
        #           activation='relu',
        #           name='hidden-layer-4')(x)
        # x = Dense(2,
        #           kernel_initializer='normal',
        #           activation='relu',
        #           name='hidden-layer-5')(x)
        # x = Dropout(rate=0.2)(x)
        output = Dense(1,
                       kernel_initializer='normal',
                       activation='linear',
                       name='output-layer')(x)
        # Model
        model = Model(inputs=input, outputs=output)
        model.summary()



        # plot_model(model,
        #            to_file=directory+'{0}.png'.format(plot_title),
        #            show_shapes=True,
        #            show_layer_names=True,
        #            rankdir='LR')

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
                            validation_data=(x_val, y_val),
                            callbacks=[tensorboard]
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
    r._create_model(5, 1, plot_title='lstm')
