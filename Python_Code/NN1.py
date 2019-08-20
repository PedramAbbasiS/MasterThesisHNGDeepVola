# Neuronal Network 1 for learning the implied vola 
# source: https://github.com/amuguruza/NN-StochVol-Calibrations/blob/master/1Factor/Flat%20Forward%20Variance/NN1Factor.ipynb 
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras import backend as K
#to do: load data
Nstrikes = 20         #????
Nmaturities = 10      #????
Nparameters = 5
Nsamples = 1000
data = np.random.rand(Nsamples,Nparameters+Nstrikes*Nmaturities)
xx = data[:,:Nparameters]
yy = data[:,Nparameters:]

#####
# split into train and test sample
X_train, X_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15, random_state=42)

#evtl. to do: scale and normalize siehe github

#Neural Network
keras.backend.set_floatx('float64')
NN1 = Sequential()
NN1.add(InputLayer(input_shape=(Nparameters,)))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(Nstrikes*Nmaturities, activation = 'linear'))
NN1.summary()

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
        
NN1.compile(loss = root_mean_squared_error, optimizer = "adam")
NN1.fit(X_train, y_train, batch_size=32, validation_data = (X_val, y_val),
        epochs = 200, verbose = True, shuffle=1)