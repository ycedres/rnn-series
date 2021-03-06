import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
# from keras.utils import plot_model
#from keras.utils import apply_modifications
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import os
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer



class MyLayer(LSTM):

    #def __init__(self, output_dim, **kwargs):
    def __init__(self, units, **kwargs):
        self.p = None
        #get_custom_objects().update({'noisy_activation_tahn': Activation(self.NHardTanhSat)})
        get_custom_objects().update({'noisy_activation_tahn': Activation(self.NTanhP)})
        super(MyLayer, self).__init__(units=units,activation='noisy_activation_tahn',**kwargs)
        #super(MyLayer, self).__init__(units=units, **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        with open('/tmp/somefile.txt', 'a') as the_file:
            the_file.write('p: {}\n'.format(input_shape))
        self.p = self.add_weight(name='p',
                                      shape=(1,10),
                                      initializer='uniform',
                                      trainable=True)
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #return K.dot(x, self.kernel)
        return super(MyLayer,self).call(x)

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], 10)
    #@staticmethod
    def NHardTanhSat(self,x,
                     c=0.25):

        HardTanh = lambda x: tf.minimum(tf.maximum(x, -1.), 1.)

        threshold = 1.001

        noise = K.random_normal(shape=K.shape(x),mean=0.0, stddev=1.0)
        # with open('/tmp/somefile.txt', 'a') as the_file:
        #     the_file.write('p: {}\n'.format(self.p))
        test = K.cast(K.greater(K.abs(x) , threshold), dtype='float32')
        res = test * HardTanh(x + c * noise) + (1. - test) * HardTanh(x) + self.p
        return res

    def NTanhP(self,x,
               use_noise=tf.constant(False, dtype=tf.bool),
               alpha=1.15,
               c=0.5,
               noise=None,
               clip_output=False,
               half_normal=False):
        """
        Noisy Hard Tanh Units: NAN with learning p
        ----------------------------------------------------
        Arguments:
            x: tensorflow tensor variable, input of the function.
            p: tensorflow variable, a vector of parameters for p.
            use_noise: bool, whether to add noise or not to the activations, this is in particular
            useful for the test time, in order to disable the noise injection.
            c: float, standard deviation of the noise
            alpha: float, the leakage rate from the linearized function to the nonlinear one.
            half_normal: bool, whether the noise should be sampled from half-normal or
            normal distribution.
        """
        HardTanh = lambda x: tf.minimum(tf.maximum(x, -1.), 1.)

        if not noise:
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)

        signs = tf.sign(x)
        delta = HardTanh(x) - x

        scale = c * (tf.sigmoid(self.p * delta) - 0.5) ** 2
        if alpha > 1.0 and half_normal:
            scale *= -1.0

        zeros = tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
        rn_noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)

        def noise_func():
            return tf.abs(rn_noise) if half_normal else zeros

        def zero_func():
            return zeros + 0.797 if half_normal else zeros

        noise = tf.cond(use_noise, noise_func, zero_func)

        res = alpha * HardTanh(x) + (1. - alpha) * x - signs * scale * noise

        if clip_output:
            return HardTanh(res)
        return res


class RLSTMNoise(object):

    def __init__(self, config=None,basedir=None,file_prefix=None,
                 input_descriptor_string=None):

        if basedir is not None:
            self._basedir = basedir
        if file_prefix is not None:
            self._file_prefix = file_prefix
        if input_descriptor_string is not None:
            self._input_descriptor_string = input_descriptor_string

        self._plot_title = 'lstm'
        self._reg = self._create_model(10, 1)
        self._batch_size = 1024
        self._epochs = 100

    def config_exp_path(self,basedir=None,file_prefix=None,
                 input_descriptor_string=None):

        self._basedir = basedir
        self._file_prefix = file_prefix
        self._input_descriptor_string = input_descriptor_string


    def configure(self,features_by_timestep):
        self._reg = self._create_model(features_by_timestep , 1)

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

        def NHardTanhSatExpectation(x,
                         use_noise=True,
                         c=0.25):

            HardTanh = lambda x: tf.minimum(tf.maximum(x, -1.), 1.)

            threshold = 1.001

            #noise = K.random_normal(shape=K.shape(x),mean=0.0, stddev=1.0)
            noise = 0.0

            test = K.cast(K.greater(K.abs(x) , threshold), dtype='float32')
            res = test * HardTanh(x + c * noise) + (1. - test) * HardTanh(x)
            return res

        print("TEEEEEEEEEEEEEEEEEEEEEEEEST")
        self._reg.layers[2].activation = NHardTanhSatExpectation
        #apply_modifications(self._reg)
        #print(self._reg.summary())
        print(type(self._reg.layers))
        [print(layer.activation) for layer in self._reg.layers if hasattr(layer,'activation')]
        x_test = features_test_set.values.astype('float32')
        x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return self._reg.predict(x_test_lstm)


    def _create_model(self,time_step, feature_by_timestep):
        # Input layer
        input = Input((time_step,
                       feature_by_timestep),
                       dtype='float32',
                       name='input-layer')
        normalize_input = BatchNormalization(name='normalize-input')(input)
        # RNN - LSTM

        from keras import backend as K
        from keras.layers import Activation
        from keras.utils.generic_utils import get_custom_objects
        import math
        import tensorflow as tf

        def relu_noise(x):

            isPositive = K.greater(x,0)

            noise = K.random_normal((K.shape(x)), mean=0.5, stddev=0.5)
             #I'm just not sure this is exactly the kind of noise you want.

            return (x * K.cast(isPositive,tf.float32)) + noise



        def custom_activation(x):
            #return (K.tanh(x)+relu_noise(x))
            return (K.tanh(x))
            #return (K.log(K.sigmoid(x)))

        def hard_tanh(x):
            return K.max(K.min(x,1),-1)

        def d(x,alpha):
            return (-1 * K.sign(x)) * K.sign(1-alpha)

        def noise_deviation(x,c,p):
            return K.square(c*(p*-0.5))

        def noisy_activation_tahn(x):
            # alpha = 0.7
            # c = 0.2
            # p = 0.3
            alpha = 0.7
            c = 0.2
            p = 0.3
            noise = K.random_normal()
            print("NOISSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
            return alpha * K.tanh(x) + (1-alpha) * hard_tanh(x) + d(x,alpha) * noise_deviation(x,c,p) * K.abs(noise)


        def NHardTanhSat(x,
                         use_noise=True,
                         c=0.25):

            HardTanh = lambda x: tf.minimum(tf.maximum(x, -1.), 1.)

            threshold = 1.001

            noise = K.random_normal(shape=K.shape(x),mean=0.0, stddev=1.0)

            test = K.cast(K.greater(K.abs(x) , threshold), dtype='float32')
            res = test * HardTanh(x + c * noise) + (1. - test) * HardTanh(x)
            return res


        #get_custom_objects().update({'noisy_activation_tahn': Activation(noisy_activation_tahn)})
        get_custom_objects().update({'noisy_activation_tahn': Activation(NHardTanhSat)})

        # x = LSTM(10,
        #          kernel_initializer='normal',
        #          #activation='noisy_activation_tahn',
        #          name='lstm-layer')(normalize_input)
        x = MyLayer(units=10,
                 kernel_initializer='normal',
                 #activation='noisy_activation_tahn',
                 name='lstm-layer')(normalize_input)
        # Fully-connect
        x = Dense(5,
                  kernel_initializer='normal',
                  activation='relu',
                  name='embudo-1')(x)
        x = Dense(10,
                  kernel_initializer='normal',
                  activation='relu',
                  name='embudo-2')(x)
        # x = Dense(5,
        #           kernel_initializer='normal',
        #           activation='relu',
        #           name='hidden-layer-2')(x)

        #x = Dropout(rate=0.2)(x)
        output = Dense(1,
                       kernel_initializer='normal',
                       activation='linear',
                       name='output-layer')(x)
        # Model
        model = Model(inputs=input, outputs=output)
        model.summary()

        # plot_model(model,
        #            to_file='{0}.png'.format(self._plot_title),
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
        print("fit NOISSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
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

        # plot_model(self._reg,
        #            to_file=directory+'{0}.png'.format(self._plot_title),
        #            show_shapes=True,
        #            show_layer_names=True,
        #            rankdir='LR')


if __name__ == '__main__':
    r = RLSTM()
    r._create_model(6, 1, plot_title='lstm')
