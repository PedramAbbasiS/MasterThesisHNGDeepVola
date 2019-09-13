# Robustness Test Jackknife
# source: https://github.com/amuguruza/NN-StochVol-Calibrations/blob/master/1Factor/Flat%20Forward%20Variance/NN1Factor.ipynb 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import KFold
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
import scipy
import random
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# functions ###################################################################
def myscale(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=(x[i] - (ub[i] + lb[i])*0.5) * 2 / (ub[i] - lb[i])
        
    return res
def myinverse(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=x[i]*(ub[i] - lb[i]) *0.5 + (ub[i] + lb[i])*0.5
        
    return res

def ytransform(y_train,y_val,y_test):
    return [scale.transform(y_train),scale.transform(y_val), 
            scale.transform(y_test)]
   
def yinversetransform(y):
    return scale.inverse_transform(y)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def root_relative_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)/y_true))   
    
def elu(x):
    #Careful function ovewrites x
    ind=(x<0)
    x[ind]=np.exp(x[ind])-1
    return x

def eluPrime(y):
    # we make a deep copy of input x
    x=np.copy(y)
    ind=(x<0)
    x[ind]=np.exp(x[ind])
    x[~ind]=1
    return x

def NeuralNetworkGradient(x):
    input1=x
    #Identity Matrix represents Jacobian with respect to initial parameters
    grad=np.eye(Nparameters)
    #Propagate the gradient via chain rule
    for i in range(NumLayers):
        input1=(np.dot(input1,NNParameters[i][0])+NNParameters[i][1])
        grad=(np.einsum('ij,jk->ik', grad, NNParameters[i][0]))
        #Elu activation
        grad*=eluPrime(input1)
        input1=elu(input1)
    #input1.append(np.dot(input1[i],NNParameters[i+1][0])+NNParameters[i+1][1])
    grad=np.einsum('ij,jk->ik',grad,NNParameters[i+1][0])
    #grad stores all intermediate Jacobians, however only the last one is used here as output
    return grad

#Cost Function for Levenberg Marquardt
def CostFuncLS(x,sample_ind):
    return (NN1.predict(x.reshape(1,Nparameters))[0]-y_test_trafo[sample_ind])
def JacobianLS(x,sample_ind):
    return NeuralNetworkGradient(x).T

def CostFunc(x,sample_ind):
    return np.sum(np.power((NN1.predict(x.reshape(1,Nparameters))[0]-y_test_trafo[sample_ind]),2))
def Jacobian(x,sample_ind):
    return 2*np.sum((NN1.predict(x.reshape(1,Nparameters))[0]-y_test_trafo[sample_ind])*NeuralNetworkGradient(x),axis=1)

###############################################################################
#plt.close("all")

#Data Import
import scipy.io
mat = scipy.io.loadmat('data_vola_24998_0005_09_11_30_240.mat')
data = mat['data_vola_clear']
data = data[:20000,:]
#Data Preperation
Nparameters = 5
S0=1.
maturities = np.array([30, 60, 90, 120, 150, 180, 210, 240])
strikes = np.array([0.9, 0.92, 0.94, 0.96, 0.98,1.0, 1.02, 1.04, 1.06, 1.08,1.1])
Nstrikes = len(strikes)   
Nmaturities = len(maturities)   
xx=data[:,:Nparameters]
yy=data[:,Nparameters:]
nsims=20
err = np.zeros((3,Nstrikes*Nmaturities))
err2 = np.zeros((nsims,3,Nstrikes*Nmaturities))

X_train_Big, X_test, y_train_Big, y_test = train_test_split(xx, yy, test_size=0.15,random_state = None)
X_train, X_val, y_train, y_val = train_test_split(X_train_Big, y_train_Big, test_size=0.15,random_state = None)
[y_train_trafo, y_val_trafo, y_test_trafo]=ytransform(y_train, y_val, y_test)
scale=StandardScaler()
y_train_transform = scale.fit_transform(y_train)
y_val_transform = scale.transform(y_val)
y_test_transform = scale.transform(y_test)
[y_train_trafo, y_val_trafo, y_test_trafo]=ytransform(y_train, y_val, y_test)
ub=np.amax(xx, axis=0)
lb=np.amin(xx, axis=0)
X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo = np.array([myscale(x) for x in X_val]) 
X_test_trafo = np.array([myscale(x) for x in X_test])
y_test_re = yinversetransform(y_test_trafo)
from random import sample 
  

#Neural Network
keras.backend.set_floatx('float64')


# neural network fit        
NN1 = Sequential()
NN1.add(InputLayer(input_shape=(Nparameters,)))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(Nstrikes*Nmaturities, activation = 'linear'))
#NN1.summary()
NN1.compile(loss = root_mean_squared_error, optimizer = "adam")
NN1.fit(X_train_trafo, y_train_trafo, batch_size=32, validation_data = (X_val_trafo, y_val_trafo),
        epochs = 200, verbose = True, shuffle=1)

prediction=[yinversetransform(NN1.predict(X_test_trafo[i].reshape(1,Nparameters))[0]) for i in range(len(X_test_trafo))]

#avg. rel error
err[0,:] = np.mean(np.abs((y_test_re-prediction)/y_test_re),axis = 0)
#std. rel. error
err[1,:] = np.std(np.abs((y_test_re-prediction)/y_test_re),axis = 0)
#max rel. error
err[2,:] = np.max(np.abs((y_test_re-prediction)/y_test_re),axis = 0)


ksresults = np.zeros((nsims,4))

for i in range(nsims):
    train_sub = sample(range(14450),int(14450*0.8))
    val_sub = sample(range(2550),int(2550*0.8))
    X_train_trafo_s = X_train_trafo[train_sub,:]
    X_val_trafo_s = X_val_trafo[val_sub,:]
    y_train_trafo_s = y_train_trafo[train_sub,:] 
    y_val_trafo_s = y_val_trafo[val_sub,:]
    # neural network fit        
    NN2 = Sequential()
    NN2.add(InputLayer(input_shape=(Nparameters,)))
    NN2.add(Dense(30, activation = 'elu'))
    NN2.add(Dense(30, activation = 'elu'))
    NN2.add(Dense(30, activation = 'elu'))
    NN2.add(Dense(Nstrikes*Nmaturities, activation = 'linear'))
    #NN1.summary()
    NN2.compile(loss = root_mean_squared_error, optimizer = "adam")
    NN2.fit(X_train_trafo_s, y_train_trafo_s, batch_size=32, validation_data = (X_val_trafo_s, y_val_trafo_s),
            epochs = 10, verbose = True, shuffle=1)
    
    prediction=[yinversetransform(NN2.predict(X_test_trafo[i].reshape(1,Nparameters))[0]) for i in range(len(X_test_trafo))]
    
    #avg. rel error
    err2[i,0,:] = np.mean(np.abs((y_test_re-prediction)/y_test_re),axis = 0)
    #std. rel. error
    err2[i,1,:] = np.std(np.abs((y_test_re-prediction)/y_test_re),axis = 0)
    #max rel. error
    err2[i,2,:] = np.max(np.abs((y_test_re-prediction)/y_test_re),axis = 0)
    weights_1 = NN1.layers[0].get_weights()[0].reshape(30*5,)
    weights_2 = NN1.layers[1].get_weights()[0].reshape(30*30,)
    weights_3 = NN1.layers[2].get_weights()[0].reshape(30*30,)
    weights_4 = NN1.layers[3].get_weights()[0].reshape(30*88,)
    weights_1p = NN2.layers[0].get_weights()[0].reshape(30*5,)
    weights_2p = NN2.layers[1].get_weights()[0].reshape(30*30,)
    weights_3p = NN2.layers[2].get_weights()[0].reshape(30*30,)
    weights_4p = NN2.layers[3].get_weights()[0].reshape(30*88,)
    ksresults[i,0] = scipy.stats.ks_2samp(weights_1,weights_1p)[1]
    ksresults[i,1] = scipy.stats.ks_2samp(weights_2,weights_2p)[1]
    ksresults[i,2] = scipy.stats.ks_2samp(weights_3,weights_3p)[1]
    ksresults[i,3] = scipy.stats.ks_2samp(weights_4,weights_4p)[1] 

err_s = np.mean(err2,axis =0)


import seaborn as sns

plt.figure(figsize=(18,5))
#==============================================================================
#max error
plt.subplot(131)
sns.distplot(err[2,:], color = 'blue', hist = False, label = 'full set')
sns.distplot(err_s[2,:], color = 'red', hist = False, label = 'reduced set')
plt.legend(prop={'size': 16})
plt.title('max errors full  vs. reduced')
plt.xlabel('max error')
plt.ylabel('Density')
#==============================================================================
#mean error
plt.subplot(132)
sns.distplot(err[0,:], color = 'blue', hist = False, label = 'full set')
sns.distplot(err_s[0,:], color = 'red', hist = False, label = 'reduced set')
plt.legend(prop={'size': 16})
plt.title('mean errors full vs. reduced')
plt.xlabel('mean error')
plt.ylabel('Density')
#==============================================================================
#std error
plt.subplot(133)
sns.distplot(err[1,:], color = 'blue', hist = False, label = 'full set')
sns.distplot(err_s[1,:], color = 'red', hist = False, label = 'reduced set')
plt.legend(prop={'size': 16})
plt.title('std errors reduced vs. reduced')
plt.xlabel('std error')
plt.ylabel('Density')

plt.savefig('RobustnessSize.png', dpi=300)

