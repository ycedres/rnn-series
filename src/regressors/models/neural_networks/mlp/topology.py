
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam

from models.neural_networks.utils.plot_utils import plot_modelTopology

def topology(config, exp_folder, load_weights=False, weights_model_fitted_path=None):
    """

    :param config:
    :param load_weights:
    :param weights_model_fitted_path:
    :return:
    """
    # -- Parameters
    n_input = int(config.get('mlp', 'n_input'))
    n_output = int(config.get('mlp', 'n_output'))
    n_hidden = list(map(lambda i: int(i), config.get('mlp', 'n_hidden').split(',')))
    dropouts = list(map(lambda i: int(i) if i == '0' else float(i),
                        config.get('mlp', 'dropouts').split(',')))
    lr = float(config.get('mlp', 'lr'))

    # - exception: if len(n_hidden) == len(dropouts): raise

    # -- topology
    inputs = Input(shape=(n_input,), name='Input-Layer')

    normalization_batch = BatchNormalization(name='normalization_batch')(inputs)

    for neurons, dropout, layer in zip(n_hidden, dropouts, range(len(n_hidden))):

        if layer == 0:
            x = Dense(neurons, activation='relu', name='Hidden_{0}'.format(layer))(normalization_batch)
            if dropout != 0:
                x = Dropout(dropout)(x)
        else:
            x = Dense(neurons, activation='relu', name='Hidden_{0}'.format(layer))(x)
            if dropout != 0:
                x = Dropout(dropout)(x)

    predictions = Dense(n_output, activation='softmax', name='Output-Layer')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()

    # plot models's topology
    plot_modelTopology(model, exp_folder, config)

    # -- Load weights: only if it selects the predict_load_model or auto
    if load_weights==True:
        model.load_weights(weights_model_fitted_path)

        return model

    # -- compile
    opt = Adam(lr=lr)
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mse'],
                  )

    return model