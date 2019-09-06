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

#Data Import
import scipy.io
mat = scipy.io.loadmat('data_price_895_0005_0875_1125_30_240.mat')
data = mat['data_price_clear']

#Data Preperation
Nparameters = 5
S0=1.
maturities = np.array([30, 60, 90, 120, 150, 180, 210, 240])
strikes = np.array([0.875,0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1,1.125])
Nstrikes = len(strikes)   
Nmaturities = len(maturities)   
xx=data[:,:Nparameters]
yy=data[:,Nparameters:]
X_train_Big, X_test, y_train_Big, y_test = train_test_split(xx, yy, test_size=0.15, random_state=42)
#Neural Network
keras.backend.set_floatx('float64')
NN1 = Sequential()
NN1.add(InputLayer(input_shape=(Nparameters,)))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(Nstrikes*Nmaturities, activation = 'linear'))
NN1.summary()

num_splits = 10
runs = -1
kf = KFold(n_splits=num_splits,random_state=42,shuffle=True)
err = np.zeros((num_splits,3,Nstrikes*Nmaturities))
test_params = np.zeros((num_splits,X_test.shape[0],Nparameters))
test_rel_errors = np.zeros((num_splits,X_test.shape[0],Nparameters))
test_median_mean = np.zeros((num_splits,2,Nparameters))
test_RMSE = np.zeros((num_splits,X_test.shape[0]))
test_RMSE_median_mean = np.zeros((num_splits,2))
for train_index, val_index in kf.split(X_train_Big,y_train_Big):
    runs += 1
    X_train, X_val = X_train_Big[train_index], X_train_Big[val_index]
    y_train, y_val = y_train_Big[train_index], y_train_Big[val_index]
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

    # neural network fit        
    NN1.compile(loss = root_mean_squared_error, optimizer = "adam")
    NN1.fit(X_train_trafo, y_train_trafo, batch_size=32, validation_data = (X_val_trafo, y_val_trafo),
            epochs = 200, verbose = True, shuffle=1)

    y_test_re = yinversetransform(y_test_trafo)
    prediction=[yinversetransform(NN1.predict(X_test_trafo[i].reshape(1,Nparameters))[0]) for i in range(len(X_test_trafo))]

    #avg. rel error
    err[runs,0,:] = np.mean(100*np.abs((y_test_re-prediction)/y_test_re),axis = 0)
    #std. rel. error
    err[runs,1,:] = 100*np.std(np.abs((y_test_re-prediction)/y_test_re),axis = 0)
    #max rel. error
    err[runs,2,:] = 100*np.max(np.abs((y_test_re-prediction)/y_test_re),axis = 0)


    # gradient methods for optimization with Levenberg-Marquardt
    NNParameters=[]
    for i in range(0,len(NN1.layers)):
        NNParameters.append(NN1.layers[i].get_weights())
    NumLayers=3
    Approx=[]
    solutions=np.zeros([num_splits,Nparameters])
    init=np.zeros(Nparameters)
    n = X_test.shape[0]
    for i in range(n):
        disp=str(np.round((i+1)/n*100,1))+"%"
        print (disp,end="\r")
        #Levenberg-Marquardt
        I=scipy.optimize.least_squares(CostFuncLS, init, JacobianLS, args=(i,), gtol=1E-10)
        solutions=myinverse(I.x)
        Approx.append(np.copy(solutions))
    LMParameters=[Approx[i] for i in range(len(Approx))]
    test_params[runs,:,:] = np.asarray(LMParameters)
    
    #Calibration Errors with Levenberg-Marquardt
    average=np.zeros([Nparameters,n])
    for u in range(Nparameters):
        for i in range(n):
            X=X_test[i][u]
            plt.plot(X,100*np.abs(LMParameters[i][u]-X)/np.abs(X),'b*')
            average[u,i]=np.abs(LMParameters[i][u]-X)/np.abs(X)
        test_median_mean[runs,0,:] = np.quantile(100*average[u,:],0.5)
        test_median_mean[runs,1,:] = np.mean(100*average[u,:])
    test_rel_errors[runs,:,:] = average.T   
    
    #RMSE
    RMSE_opt = np.zeros(n)
    Y = len(y_test[0,:])
    for i in range(n):
        Y = y_test[i,:]
        Y_pred = yinversetransform(NN1.predict(myscale(LMParameters[i]).reshape(1,Nparameters))[0])
        RMSE_opt[i] = np.sqrt(np.mean((Y-Y_pred)**2))
    test_RMSE_median_mean[runs,:] = [np.quantile(100*RMSE_opt,0.5),np.mean(100*RMSE_opt)]
    test_RMSE[runs,:] = RMSE_opt 
  