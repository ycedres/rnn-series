# ********************************************************************
#
#                             CALLBACKS
#
# *********************************************************************
import os

# -- scikit
from sklearn.metrics.regression import mean_absolute_error
from sklearn.metrics.regression import mean_squared_error

# -- keras
from keras.callbacks import Callback


class MetricCallBack(Callback):
    """
    Regressors callbacks
    """
    def __init__(self,
                 exp_path,
                 training_data,
                 validation_data,
                 predict_bach_size=1024,
                 include_on_batch=False):
        super(MetricCallBack, self).__init__()
        self.exp_path = exp_path
        self.x_train = training_data[0]
        self.y_train = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.predict_bach_size = predict_bach_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        """
        Validation of train and validation set after fitting all batches.
        For each epoch the model does this operation.

        :param batch: batch of the experiemnt.
        :param logs: dictionary with the log of each epoch.
        :return: valida
        """
        if self.include_on_batch:
            pass

    def on_train_begin(self, logs={}):
        """
        Include e new metric in the compilation of the model.

        :param logs: ictionary with the log of each epoch.
        :return:  include e new metric in the compilation of the model.
        """
        regressor_metrics = [
            #'bias',
            'mae',
            'mse',
            #'rmse',
            #'sde'
        ]
        for metric in regressor_metrics:
            if not (metric in self.params['metrics']):
                self.params['metrics'].append(metric)

            if not ('{0}_val'.format(metric) in self.params['metrics']):
                self.params['metrics'].append('{0}_val'.format(metric))

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        """
        For each epoch calculates the roc-auc of the train and validation set, and
        save the model (topology + compilation).
        :param epoch: epochs
        :param logs: logs dictionary where include the values of roc-auc for each epoch.
        :return: a model saved and the roc-auc for train and validation set for each epoch.
        """

        # train set
        y_prediction = self.model.predict(self.x_train, batch_size=self.predict_bach_size)
        logs['mae'] = mean_absolute_error(self.y_train, y_prediction)
        logs['mse'] = mean_squared_error(self.y_train, y_prediction)

        if self.validation_data:
            y_prediction_val = self.model.predict(self.x_val,
                                                  batch_size=self.predict_bach_size)
            logs['mae_val'] = mean_absolute_error(self.y_val, y_prediction_val)
            logs['mse_val'] = mean_squared_error(self.y_val, y_prediction_val)

        save_model = os.path.join('saved_model_epoch_{0}'.format(epoch))

        self.model.save(save_model)


        return