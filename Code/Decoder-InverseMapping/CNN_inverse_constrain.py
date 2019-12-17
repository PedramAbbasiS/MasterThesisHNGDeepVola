"""
Created on Thu Nov 28 13:27:14 2019

@author: Henrik Brautmeier

CNN for decoding!
"""

# Neuronal Network 1 for learning the implied vola 
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential,Model
from keras.layers import InputLayer,Dense,Flatten, Conv2D, Dropout, Input,ZeroPadding2D,MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import py_vollib.black_scholes.implied_volatility as vol
#import time
import scipy
import scipy.io

# Data Import

mat         = scipy.io.loadmat('data_vola_maxbounds_50000_0005_09_11_30_210.mat')
data        = mat['data_vola']
Nparameters = 5
maturities  = np.array([30, 60, 90, 120, 150, 180, 210])
strikes     = np.array([0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1])
Nstrikes    = len(strikes)   
Nmaturities = len(maturities)   
xx          = data[:,:Nparameters]
yy          = data[:,Nparameters+2:]


# split into train and test sample
X_train, X_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.15)#, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15)#, random_state=42)

Ntest= X_test.shape[0]
Ntrain= X_train.shape[0]
Nval= X_val.shape[0]
def ytransform(y_train,y_val,y_test):
    #return [scale.transform(y_train),scale.transform(y_val), 
    #        scale.transform(y_test)]
    return [y_train,y_val,y_test]
def yinversetransform(y):
    return y
    #return scale.inverse_transform(y)
[y_train_trafo, y_val_trafo, y_test_trafo]=ytransform(y_train, y_val, y_test)
y_train_trafo = np.asarray([y_train[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
y_val_trafo =  np.asarray([y_val[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])
y_test_trafo =  np.asarray([y_test[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])
ub=np.amax(xx, axis=0)
lb=np.amin(xx, axis=0)
def myscale(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=(x[i] - (ub[i] + lb[i])*0.5) * 2 / (ub[i] + lb[i])
    return res
def myinverse(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=x[i]*(ub[i] + lb[i]) *0.5 + (ub[i] + lb[i])*0.5
    return res
def inverse_loss(x):
    res = K.zeros((Nparameters,))
    for i in range(Nparameters):
        res[i]=x[i]*(ub[i] + lb[i]) *0.5 + (ub[i] + lb[i])*0.5
    return res

X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = np.array([myscale(x) for x in X_test])
X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = X_test_trafo.reshape((Ntest,5,1,1))
X_train_trafo = X_train_trafo.reshape((Ntrain,5,1,1))
X_val_trafo   = X_val_trafo.reshape((Nval,5,1,1))
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))   
def root_relative_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))    
def rel_mse_constraint(y_true, y_pred):
        trafo = inverse_loss(y_pred)
        constraint = np.asarray([trafo[i,1]*(trafo[i,3])**2+trafo[i,2] for i in range(K.shape(trafo)[0])])
        return K.mean(K.square((y_pred - y_true)/y_true))+np.mean(constraint>=1) 
#Neural Network
keras.backend.set_floatx('float64')

y_train_trafo2 = y_train_trafo.reshape((Ntrain,Nmaturities,Nstrikes,1))
y_test_trafo2 = y_test_trafo.reshape((Ntest,Nmaturities,Nstrikes,1))
y_val_trafo2 = y_val_trafo.reshape((Nval,Nmaturities,Nstrikes,1))
X_val_trafo2 = X_val_trafo.reshape((Nval,Nparameters))
X_train_trafo2 = X_train_trafo.reshape((Ntrain,Nparameters))
X_test_trafo2 = X_test_trafo.reshape((Ntest,Nparameters))
"""
input1 = Input(shape = (7,9,1))
x1 = Conv2D(64, kernel_size=3, activation='relu')(input1)
x2 = Conv2D(64, kernel_size=3, activation='relu')(x1)
x3 = Flatten()(x2)
x4 = Dense(50, activation = 'elu')(x3)
seq1 = Dense(1, activation = 'linear',use_bias=True)(x4)
seq2 = Dense(1, activation = 'linear',use_bias=True)(x4)
seq3 = Dense(1, activation = 'linear',use_bias=True)(x4)
seq4 = Dense(1, activation = 'linear',use_bias=True)(x4)
seq5 = Dense(1, activation = 'linear',use_bias=True)(x4)
out1 = keras.layers.merge.concatenate([seq1, seq2, seq3,seq4,seq5], axis=-1)
NN2 = Model(inputs=input1, outputs=out1)
"""
NN2 = Sequential() 
NN2.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2.add(Conv2D(64,(3, 3),padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(Conv2D(64,(2, 2),padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(MaxPooling2D(pool_size=(2, 2)))
NN2.add(Conv2D(64,(2, 2),padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(Conv2D(64,(2, 2),padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),padding='valid',strides =(1,1),activation ='tanh'))

NN2.add(Flatten())
NN2.add(Dense(5,activation = 'linear',use_bias=True))
NN2.summary()
#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN2.compile(loss =rel_mse_constraint, optimizer = "adam",metrics=["MAPE", "MSE"])

NN2.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2),
        epochs = 50, verbose = True, shuffle=1)



prediction = NN2.predict(y_test_trafo2)
