import os
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt

def crossValidation(data, percentage):
    dataTrainSize = int(round((percentage/100.0)*len(data)))
    dataTrain = data[0:dataTrainSize]
    dataTest = data[dataTrainSize:]
    return dataTrain, dataTest

def normalization(X):
    return X / np.max(X)

def normalization_maxMin(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def preproceso_presistencia(X):
    """
    Preparacion datos para entrenamiento sin ventana
    :return:
    """
    feature = X[:-1]
    label = X[1:]
    return feature, label

def preproceso_window(X, window_size=10, horizon=1):
    """
    Preparacion datos para entrenar con ventana
    :param X:
    :param window_size:
    :param horizon:
    :return:
    """
    x_data = []; y_data = []
    horizon = horizon - 1
    for i in range(len(X) - window_size - horizon):
        x_data.append(X[i:i+window_size])
        y_data.append(X[i+window_size + horizon])
    return np.array(x_data), np.array(y_data)


def plot_serie(dataTrain, dataTest,
               title='Train and Test Dataset',
               savefig=False,
               namePlot=None):
    plt.figure(figsize = (15, 5))
    plt.plot(dataTrain, lw=2, label="Train")
    plt.plot(dataTest, lw=2, label="Test")
    plt.title(title); plt.ylabel('m/sg')
    plt.legend()
    if savefig == True:
        if os.path.exists('../plots') == 0: os.mkdir('../plots')
        plt.savefig(namePlot)
        shutil.copy('{0}.png'.format(namePlot), '../plots')
        os.remove('{0}.png'.format(namePlot))
    if savefig == False: plt.show()

def plot_serie_pred(dataTest, pred,
                    title='Train and Test Dataset',
                    savefig=False,
                    namePlot=None):
    plt.figure(figsize = (15, 5))
    plt.plot(range(len(dataTest)), dataTest, lw=2, label="Real")
    plt.plot(range(len(pred)), pred, '-', lw=2, label="Predicted")
    plt.title(title); plt.ylabel('m/sg')
    plt.legend()
    if savefig == True:
        if os.path.exists('../plots') == 0: os.mkdir('../plots')
        plt.savefig(namePlot)
        shutil.copy('{0}.png'.format(namePlot), '../plots')
        os.remove('{0}.png'.format(namePlot))
    if savefig == False: plt.show()


class Error(object):

    def __int__(self, X, Y):
        self.X = X
        self.Y = Y

    @staticmethod
    def bias(X, Y):
        '''
        X = y_true
        Y = y_pred
        '''
        # poner que X tiene que se un array
        return np.sum(X - Y)/len(X)

    @staticmethod
    def mae(X, Y):
        return np.sum(abs(X - Y)) / len(X)

    @staticmethod
    def mse(X, Y):
        return np.sum(np.power(X -Y, 2)) / len(X)

    @staticmethod
    def rmse(X, Y):
        return np.sqrt(np.sum(np.power((X-Y), 2))) / len(X)

    @staticmethod
    def sde(X, Y):
        #return math.sqrt(np.sum(np.power((BIAS - eBIAS), 2)))
        pass

    @staticmethod
    def smape(X, Y):
        return np.sum(abs(Y - X) / (abs(Y) + abs(X))*0.5) / len(X)