
import pandas as pd

from utils import *

# --- Cargamos los datos
dataSet = pd.read_csv('../Database/vViento.csv', names=['date', 'time', 'viento'],
                      delimiter=';')

# --- Cross Validation
dataTrain, dataTest = crossValidation(dataSet['viento'], 80)

# --- Noramlizamos
dataTrain = normalization(dataTrain)
dataTest = normalization(dataTest)

# --- Prepoceso para entrenamiento
X_train, X_test = prepoceso_window(dataTrain)

# --- CUIDADO que es un DataFrame
print(type(X_train[0].values))