import time
import pandas as pd

from utils import *
from model.knearestPrediction import KNeighbors
from model.svmPrediction import svm

# --- Cargamos los datos
dataSet = pd.read_csv('../Database/vViento_rolling_mean_250.csv', names=['viento'],
                      delimiter=';')


# --- Cross Validation
dataTrain, dataTest = crossValidation(dataSet['viento'], 80)

plot_serie(dataTrain, dataTest,
           title='Train and Test data: Serie de viento con rolling mean f=250\nNoramalizacion [0,1]',
           savefig=True,
           namePlot='serie_viento_rolling_mean_f250'
           )

# --- Prepoceso para entrenamiento: shape=1 --> ?
X_train, Y_train = preproceso_window(dataTrain.values)
X_test, Y_test = preproceso_window(dataTest.values)

for i in range(1):
    print(X_train[i], '-->', Y_train[i])

#if len(dataTrain.shape) != 2: dataTrain = dataTrain[:, np.newaxis]
#print(dataTrain.shape)

# *****************************************************************************************
#
# --- Entrenamos con k-Neaighbors
#
# *****************************************************************************************
"""
print("*"*50)
print("********", "KNeares", "********")
print("*"*50)

t1 = time.time()
pred, scores = KNeighbors(X_train, Y_train, X_test, Y_test, weights=['uniform'])
t2 = time.time()
CPU_time = t2 - t1

#print(scores)
#print(len(pred))


# Plot prediction -----------------------------------------------
plt.figure(5, figsize = (15, 5))
plt.plot(range(len(dataTest)), dataTest, lw=2, label="Test")
for config, weight in zip(pred,['uniform', 'distance']) :
    plt.plot(range(len(config)), config, lw=2, label=weight)
plt.title('Predicted vs. Test')
plt.legend()
if os.path.exists('../plots') == 0: os.mkdir('../plots')
namePlot = 'knearest_serie_viento_rolling_mean_f250_pred_test'
plt.savefig(namePlot)
shutil.copy('{0}.png'.format(namePlot), '../plots')
os.remove('{0}.png'.format(namePlot))



if os.path.exists('../model/knearestTrainResults') == 0: os.mkdir('../model/knearestTrainResults')
f1 = open("../model/knearestTrainResults" + "/" + "error.txt", 'a')
for i in scores: f1.write(str(i) + '\n')
f1.write('CPU_Time' + ';' + 'null' + ';' + str(CPU_time))
f1.close()
"""

# *****************************************************************************************
#
# --- Entrenamos SVM
#
# *****************************************************************************************

print("*"*50)
print("********", "SVM", "********")
print("*"*50)

kernels = ['linear']

pred, scores = svm(X_train, Y_train, X_test, Y_test, kernels=kernels)
#print(scores)

# Plot prediction -----------------------------------------------
plt.figure(6, figsize = (15, 5))
plt.plot(range(len(dataTest)), dataTest, lw=2, label="Test")
for config, kernel in zip(pred,kernels) :
    plt.plot(range(len(config)), config, lw=2, label=kernel)
plt.title('Predicted vs. Test')
plt.legend()
if os.path.exists('../plots') == 0: os.mkdir('../plots')
namePlot = 'svn_serie_viento_rolling_mean_f250_pred_test'
plt.savefig(namePlot)
shutil.copy('{0}.png'.format(namePlot), '../plots')
os.remove('{0}.png'.format(namePlot))


if os.path.exists('../model/svnTrainResults') == 0: os.mkdir('../model/svnTrainResults')
f1 = open("../model/svnTrainResults" + "/" + "error.txt", 'a')
for i in scores: f1.write(str(i) + '\n')
f1.close()
