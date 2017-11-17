
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

from models.neural_networks.utils.plot_utils import plot_modelTopology


def lstm_topology(config, exp_folder):

    print ('I am at lstm_topology')

    time_steps = 5
    features_by_timesteps = 15
    n_units = 20
    dropout = 0.4
    recurrent_dropout = 0.4
    lr = 0.01

    # -- topology
    inputs = Input((time_steps, features_by_timesteps),
                    dtype='float32', name='Input-Layer')

    timestep_fullyconnect = TimeDistributed(Dense(features_by_timesteps,
                                                  activation='linear'),
                                            name='timestep_fullyconnect')(inputs)

    x = LSTM(n_units,
             dropout=dropout,
             return_sequences=False,
             recurrent_dropout=recurrent_dropout)(timestep_fullyconnect)

    # fully-connect layers between last LSTM cell and the output layer


    # output layer
    outputs = Dense(1, name='Output-Layer')(x)

    model = Model(inputs, outputs)

    model.summary()

    # plot model's topology
    plot_modelTopology(model, exp_folder, config)

    opt = Adam(lr=lr)

    #model.compile(lost='mean_squared_error',
    #              optimizer=opt,
    #              metrics=['mae'])


    return model
