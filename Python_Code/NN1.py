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
#to do: load data
data=np.load('rBergomiTrainSet.txt')
strikes=np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ])
maturities=np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ])
Nparameters = 4
Nstrikes = len(strikes)   
Nmaturities = len(maturities)   
xx=data[:,:Nparameters]
yy=data[:,Nparameters:]

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
[y_train, y_val, y_test]=ytransform(y_train, y_val, y_test)

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

X_train = np.array([myscale(x) for x in X_train])
X_val = np.array([myscale(x) for x in X_val])
X_test = np.array([myscale(x) for x in X_test])

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


S0=1.
y_test_re = yinversetransform(y_test)
prediction=[yinversetransform(NN1.predict(X_test[i].reshape(1,Nparameters))[0]) for i in range(len(X_test))]
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