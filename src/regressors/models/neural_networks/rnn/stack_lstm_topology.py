from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

from models.neural_networks.utils.plot_utils import plot_modelTopology

def stack_lstm_topology(config, exp_folder):

    print('I am at slack_lstm_topology')

    time_steps = 5
    features_by_timesteps = 15
    n_units = 20
    dropout = 0.4
    recurrent_dropout = 0.4
    lr = 0.01
    n_units = [40, 30, 20]
    dropouts = [0.8, 0.8, 0.8]
    recurrent_dropouts = [0.8, 0.8, 0.8]

    # -- topology
    inputs = Input((time_steps, features_by_timesteps),
                   dtype='float32', name='Input-Layer')

    timestep_fullyconnect = TimeDistributed(Dense(features_by_timesteps,
                                                  activation='linear'),
                                            name='timestep_fullyconnect')(inputs)


    for n_unit, dropout, recurrent_dropout, stack in zip(n_units, dropouts, recurrent_dropouts, range(len(n_units))):

        print(n_unit, dropout, recurrent_dropout, stack)

        if stack == 0:
            x = LSTM(n_unit,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout)(timestep_fullyconnect)
        else:
            x = LSTM(n_unit,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout)(x)

    # output layer
    outputs = Dense(1, name='Output-Layer')(x)

    model = Model(inputs, outputs)

    model.summary()

    # plot models's topology
    plot_modelTopology(model, exp_folder, config)

    #opt = Adam(lr=lr)

    #models.compile(lost='mean_squared_error',
    #              optimizer=opt,
    #              metrics=['mae'])

    return model
