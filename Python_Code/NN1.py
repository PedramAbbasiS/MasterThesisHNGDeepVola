# Neuronal Network 1 for learning the implied vola 
# source: https://github.com/amuguruza/NN-StochVol-Calibrations/blob/master/1Factor/Flat%20Forward%20Variance/NN1Factor.ipynb 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import py_vollib.black_scholes.implied_volatility as vol
#S = 1
#flag = 'c'
#r = .0
#to do: load data
#datatest =np.load('rBergomiTrainSet.txt')
#y_testdata=data[:,Nparameters:]
#print(np.mean(y_testdata))
#print(np.min(y_testdata))
#print(np.max(y_testdata))
#strikes=np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ])
#maturities=np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ])
#data = np.load('data_shortmat_MC_1e5.npy')
data = np.load('data_IV_1e4_MC_1e4.npy')
#data1 = data[(data!=0).all(axis=1)]
#np.count_nonzero(data1)
Nparameters = 5
maturities = np.array([20, 50, 80, 110, 140])
strikes = np.array([0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1])
Nstrikes = len(strikes)   
Nmaturities = len(maturities)   
xx=data[:,:Nparameters]
yy=data[:,Nparameters+1:]

#yy = np.full_like(y_p, 0.0)
#stri = np.zeros((len(maturities), len(strikes)))
#for i in range(len(y_p)):
#    price_new = y_p[i,:].reshape((len(maturities), len(strikes)))
#    for m in range(len(maturities)):
#        for k in range(len(strikes)):
#            stri[m,k] = vol.implied_volatility(price_new[m,k], S, strikes[k], maturities[m]/252, r, flag)
#    yy[i] =  stri.reshape((1,len(maturities)*len(strikes)))   


#yy=data[:,Nparameters:]

#Nstrikes = 20         #????
#Nmaturities = 10      #????
#Nparameters = 5
#Nsamples = 1000
#data = np.random.rand(Nsamples,Nparameters+Nstrikes*Nmaturities)
#xx = data[:,:Nparameters]
#yy = data[:,Nparameters:]

#####
# split into train and test sample
X_train, X_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15, random_state=42)

#evtl. to do: scale and normalize siehe github
scale=StandardScaler()
y_train_transform = scale.fit_transform(y_train)
y_val_transform = scale.transform(y_val)
y_test_transform = scale.transform(y_test)
def ytransform(y_train,y_val,y_test):
    return [scale.transform(y_train),scale.transform(y_val), 
            scale.transform(y_test)]
   
def yinversetransform(y):
    return scale.inverse_transform(y)

[y_train_trafo, y_val_trafo, y_test_trafo]=ytransform(y_train, y_val, y_test)

ub=np.amax(xx, axis=0)
lb=np.amin(xx, axis=0)
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

X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo = np.array([myscale(x) for x in X_val])
X_test_trafo = np.array([myscale(x) for x in X_test])

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
NN1.fit(X_train_trafo, y_train_trafo, batch_size=32, validation_data = (X_val_trafo, y_val_trafo),
        epochs = 200, verbose = True, shuffle=1)


test = yinversetransform(NN1.predict(X_test_trafo))
y_test_trans = yinversetransform(y_test_trafo)
error = np.abs((test - y_test_trans))/np.abs(y_test_trans)
error1 = np.zeros((len(error[0,:])))
for i in range(len(error[0,:])):
    error1[i] = np.mean(error[:,i])
    
error2 = np.zeros((len(t[0,:])))
for i in range(len(t[0,:])):
    error2[i] = np.mean(t[:,i])

S0=1.
y_test_re = yinversetransform(y_test_trafo)
prediction=[yinversetransform(NN1.predict(X_test_trafo[i].reshape(1,Nparameters))[0]) for i in range(len(X_test_trafo))]
plt.figure(1,figsize=(14,4))
ax=plt.subplot(1,3,1)
err = np.mean(100*np.abs((y_test_re-prediction)/y_test_re),axis = 0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err.reshape(Nmaturities,Nstrikes))
plt.colorbar(format=mtick.PercentFormatter())

ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)

ax=plt.subplot(1,3,2)
err = 100*np.std(np.abs((y_test_re-prediction)/y_test_re),axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err.reshape(Nmaturities,Nstrikes))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)

ax=plt.subplot(1,3,3)
err = 100*np.max(np.abs((y_test_re-prediction)/y_test_re),axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err.reshape(Nmaturities,Nstrikes))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
plt.tight_layout()
plt.savefig('rBergomiNNErrors.png', dpi=300)
plt.show()