
import os

from keras.utils import plot_model

def plot_modelTopology(model, exp_path, config):
    """
    Plot the topology of the network in PNG format.
    :param model: keras¡ object
    :param exp_path: path of the folder experiemnt.
    :param config: ConfigParser's object
    :return: Plot of the topology of the neural network.
    """

    exp_name = config.get('path_exp', 'exp_name')
    plot_path = os.path.join(exp_path, exp_name)

    plot_model(model,
               to_file='{0}.png'.format(plot_path),
               show_shapes=True,
               show_layer_names=True,
               rankdir='LR')