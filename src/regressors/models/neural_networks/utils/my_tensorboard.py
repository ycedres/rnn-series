
import os
import tensorflow as tf

from keras.callbacks import Callback
from keras import backend as K

class MyTensorboard(Callback):
    """
    Tensorboard
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True):

        super(MyTensorboard).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')

        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.batch_size = batch_size
        self.write_graph = write_graph

    def set_model(self, model):
        """
        :param model: keras model object
        :return: three folders inside the experiment folder. Each of them will have the
                 simulation data for the train, validation and test set.
        """
        if self.write_graph:
            pass
        else:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'training'))
            self.val_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'validation'))
            self.test_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'test'))

    def on_epoch_end(self, epoch, logs=None):
        """
        In each epoch the model save the metrics in a tensorboard graph.
        If the user want to see the tensorboard: tensorboard --logdir=path/to/log-directory

        :param epoch: epoch of the model
        :param logs: dict with the values of each metric.
        :return: the log of the train, validation and test set inside the folder created in
                 the python method 'set_model'.
        """
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            if name[0:3] == 'val':
                val_name =name[4:]
                val_value = value

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = val_value.item()
                summary_value.tag = val_name
                self.val_writer.add_summary(summary, epoch)
            else:
                train_name = name
                train_value = value

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = train_value.item()
                summary_value.tag = train_name
                self.train_writer.add_summary(summary, epoch)

            self.train_writer.flush()
            self.val_writer.flush()

    def on_train_end(self, _):
        """
        Close the tensorflow FileWriter
        """
        self.train_writer.close()
        self.val_writer.close()
        self.test_writer.close()

