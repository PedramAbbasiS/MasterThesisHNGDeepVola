#!/usr/bin/env python
# coding: utf-8

# # Summary
# __The Calibration Network / Inverse Mapping penalises now constraint violations!__
# 
# 
# This code gives a short summary on the current progress (last update: 20.01.2020).
# In the following a CNN aswell as a FFNN are trained to learn the mapping HNG-Parameters (a,b,g*,w,h0) to HNG-Volatility surface. A first approach on  training the inverse mapping with CNN is given. 
# For the used dataset (50.000 szenarios), each szenario is generated as followed:
# S&P500 returns are used to get optimal HNG parameters for each week in 2015 (10years rolling window). The initial variance is used as variable aswell.
# Afterwards for each week in 2015 Call Prices are used to get optimal HNG parameters. The initial variance is set to the corrensponding value from MLE optimization.
# Those 52 optimal parameters combinations are then used as bounds. 
# To generate a szenario a random value between the bounds is uniformly choosen (initial variance included)
# and a grid of implied variance is calculated for Maturities [30, 60, 90, 120, 150, 180, 210] days and Moneyness     [0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1].
# 

# # 1. Preambel and Data Import

# In[2]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[3]:


# Preambel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import InputLayer,Dense,Flatten, Conv2D, Dropout, Input,ZeroPadding2D,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import py_vollib.black_scholes.implied_volatility as vol
#import time
import scipy
import scipy.io
import tensorflow as tf

# scaler
def myinverse(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=x[i]*(ub[i] - lb[i]) *0.5 + (ub[i] + lb[i])*0.5
    return res

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))   
def root_relative_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))   

def rmse_constraint(param):
    def rel_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))  +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint
def mse_constraint(param):
    def rel_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.mean(K.square(y_pred - y_true)) +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint
    
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

ub=np.amax(xx, axis=0)
lb=np.amin(xx, axis=0)
diff = ub-lb
bound_sum =ub+lb

scale = StandardScaler()

def ytransform(y_train,y_val,y_test):
    return [scale.transform(y_train),scale.transform(y_val), 
            scale.transform(y_test)]
    return [y_train,y_val,y_test]

def yinversetransform(y):
    return y
    return scale.inverse_transform(y)
    
def myscale(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=(x[i] - (ub[i] + lb[i])*0.5) * 2 / (ub[i] - lb[i])
    return res


# split into train and test sample
X_train, X_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.15)#, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15)#, random_state=42)

y_train_transform = scale.fit_transform(y_train)
y_test_transform = scale.transform(y_test)
y_val_transform = scale.transform(y_val)

Ntest= X_test.shape[0]
Ntrain= X_train.shape[0]
Nval= X_val.shape[0]
tf.keras.backend.set_floatx('float64')  
es = EarlyStopping(monitor='val_loss', mode='min', verbose=True, restore_best_weights=True,patience = 20)


# # 2. CNN as Encoder / Pricing Kernel

# In[4]:


# reshaping train/test sets for structure purposes
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config=config)
set_session(sess)

[y_train_trafo, y_val_trafo, y_test_trafo]=ytransform(y_train, y_val, y_test)
y_train_trafo = np.asarray([y_train[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
y_val_trafo =  np.asarray([y_val[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])
y_test_trafo =  np.asarray([y_test[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])

X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = np.array([myscale(x) for x in X_test])
X_train_trafo = np.array([myscale(x) for x in X_train])
X_val_trafo   = np.array([myscale(x) for x in X_val])
X_test_trafo  = X_test_trafo.reshape((Ntest,5,1,1))
X_train_trafo = X_train_trafo.reshape((Ntrain,5,1,1))
X_val_trafo   = X_val_trafo.reshape((Nval,5,1,1))


# In[5]:


NN1 = Sequential() 
NN1.add(InputLayer(input_shape=(Nparameters,1,1,)))
NN1.add(ZeroPadding2D(padding=(2, 2)))
NN1.add(Conv2D(32, (3, 1), padding='valid',use_bias =True,strides =(1,1),activation='elu'))#X_train_trafo.shape[1:],activation='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(1,1),activation ='elu'))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (3,3),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
NN1.add(ZeroPadding2D(padding=(1,1)))
NN1.add(Conv2D(32, (2, 2),padding='valid',use_bias =True,strides =(2,1),activation ='elu'))
#NN1.add(MaxPooling2D(pool_size=(2, 1)))
#NN1.add(Dropout(0.25))
#NN1.add(ZeroPadding2D(padding=(0,1)))
NN1.add(Conv2D(Nstrikes, (2, 1),padding='valid',use_bias =True,strides =(2,1),activation ='linear', kernel_constraint = tf.compat.v1.keras.constraints.NonNeg()))
#NN1.add(MaxPooling2D(pool_size=(4, 1)))
NN1.summary()

#NN1.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE","MSE"])
NN1.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN1.fit(X_train_trafo, y_train_trafo, batch_size=64, validation_data = (X_val_trafo, y_val_trafo),
        epochs = 200, verbose = True, shuffle=1,callbacks=[es])

NN1.save_weights("scalingNN1.h5")
# ### 2.1 Results 
# The following plots show the performance on the testing set
# 

# In[9]:


#error plots
S0=1.
y_test_re    = yinversetransform(y_test_trafo).reshape((Ntest,Nmaturities,Nstrikes))
prediction   = yinversetransform(NN1.predict(X_test_trafo)).reshape((Ntest,Nmaturities,Nstrikes))
err_rel_mat  = np.zeros(prediction.shape)
err_mat      = np.zeros(prediction.shape)
for i in range(Ntest):
    err_rel_mat[i,:,:] =  np.abs((y_test_re[i,:,:]-prediction[i,:,:])/y_test_re[i,:,:])
    err_mat[i,:,:] =  np.square((y_test_re[i,:,:]-prediction[i,:,:]))
idx = np.argsort(np.max(err_rel_mat,axis=tuple([1,2])), axis=None)

#bad_idx = idx[:-200]
bad_idx = idx
#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
ax=plt.subplot(2,3,1)
err1 = 100*np.mean(err_rel_mat[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(err_rel_mat[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(err_rel_mat[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(err_mat[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(err_mat[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(err_mat[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()
plt.savefig('scalingNN1.png', dpi=300)


# In[12]:


#import dill
#dill.dump_session('notebook_env.db')

#dill.load_session('notebook_env.db')


# # 3. CNN as  Decoder/Inverse Mapping / Calibration

# In[13]:


# reshaping for cnn purposes
y_train_trafo2 = y_train_trafo.reshape((Ntrain,Nmaturities,Nstrikes,1))
y_test_trafo2 = y_test_trafo.reshape((Ntest,Nmaturities,Nstrikes,1))
y_val_trafo2 = y_val_trafo.reshape((Nval,Nmaturities,Nstrikes,1))
X_val_trafo2 = X_val_trafo.reshape((Nval,Nparameters))
X_train_trafo2 = X_train_trafo.reshape((Ntrain,Nparameters))
X_test_trafo2 = X_test_trafo.reshape((Ntest,Nparameters))
""" old structure with multi-purpose network 
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


# In[ ]:


NN2 = Sequential() 
NN2.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(MaxPooling2D(pool_size=(2, 2)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(ZeroPadding2D(padding=(1,1)))
NN2.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2.add(Flatten())
NN2.add(Dense(5,activation = 'linear',use_bias=True))
NN2.summary()


#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN2.compile(loss =mse_constraint(0.25), optimizer = "adam",metrics=["MAPE", "MSE"])
history = NN2.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2),
    epochs=200, verbose = True, shuffle=1,callbacks=[es])

NN2.save_weights("scalingNN2.h5")
# ### 3.1 Results
# Take care these results are on scaled parameter values and not rescaled yet!

# In[ ]:


def constraint_violation(x):
    return np.sum(x[:,0]*x[:,2]**2+x[:,1]>=1)/x.shape[0],x[:,0]*x[:,2]**2+x[:,1]>=1,x[:,0]*x[:,2]**2+x[:,1]
prediction = NN2.predict(y_test_trafo2)

prediction_invtrafo= np.array([myinverse(x) for x in prediction])

#prediction = NN2.predict(y_test_trafo2)
prediction_std = np.std(prediction_invtrafo,axis=0)
error = np.zeros((Ntest,Nparameters))
for i in range(Ntest):
    error[i,:] =  np.abs((X_test_trafo2[i,:]-prediction_invtrafo[i,:])/X_test_trafo2[i,:])
err1 = np.mean(error,axis = 0)
err2 = np.median(error,axis = 0)
err_std = np.std(error,axis = 0)
idx = np.argsort(error[:,0], axis=None)
good_idx = idx[:-100]


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
_,_,c =constraint_violation(prediction_invtrafo)
_,_,c2 =constraint_violation(X_test)


testing_violation = c>=1
testing_violation2 = (c<1)
vio_error = error[testing_violation,:]
vio_error2 = error[testing_violation2,:]


plt.figure(figsize=(14,4))
ax=plt.subplot(1,3,1)
plt.boxplot(error)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors")
ax=plt.subplot(1,3,2)
plt.boxplot(vio_error)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors parameter violation")
ax=plt.subplot(1,3,3)
plt.boxplot(vio_error2)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors no parameter violation")
plt.show()
plt.savefig('Ultra_NN2_1.png', dpi=300)

print("violation error mean in %:",100*np.mean(vio_error,axis=0))
print("no violation error mean in %:",100*np.mean(vio_error2,axis=0))
print("violation error median in %:",100*np.median(vio_error,axis=0))
print("no violation error median in %:",100*np.median(vio_error2,axis=0))
print("error mean in %:",100*err1)
print("error median in %:",100*err2)

fig = plt.figure()
plt.scatter(c2,c)
plt.plot(np.arange(0, np.max(c),0.5),np.arange(0, np.max(c),0.5),'-r')
plt.xlabel("True Constraint")
plt.ylabel("Forecasted Constraint")

plt.savefig('Ultra_NN2_2.png', dpi=300)

plt.figure(figsize=(14,4))
ax=plt.subplot(1,5,1)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
plt.scatter(c2[testing_violation],vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,2)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
plt.scatter(c2[testing_violation],vio_error[:,1],c="b",s=1,marker="x",label="beta con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,3)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
plt.scatter(c2[testing_violation],vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,4)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
plt.scatter(c2[testing_violation],vio_error[:,3],c="b",s=1,marker="x",label="omega con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,5)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,4],c="r",s=1,marker="x",label="sigma0 no con")
plt.scatter(c2[testing_violation],vio_error[:,4],c="b",s=1,marker="x",label="sigma0 con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
plt.savefig('Ultra_NN2_5.png', dpi=300)
fig = plt.figure()
plt.scatter(c2,c)
plt.plot(np.arange(0, np.max(c),0.5),np.arange(0, np.max(c),0.5),'-r')
plt.xlabel("True Constraint")
plt.ylabel("Forecasted Constraint")
plt.savefig('Ultra_NN2_3.png', dpi=300)

plt.figure(figsize=(14,4))
ax=plt.subplot(1,5,1)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,2)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,1],c="b",s=1,marker="x",label="beta con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,3)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,4)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,3],c="b",s=1,marker="x",label="omega con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,5)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,4],c="r",s=1,marker="x",label="sigma0 no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,4],c="b",s=1,marker="x",label="sigma0 con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()

plt.savefig('Ultra_NN2_4.png', dpi=300)


# # 3.2 Testing the performace of the AutoEncoder/Decoder Combination
# We test how the two previously trained NN work together. First, HNG-Vola surfaces are used to predict the underlying parameters with NN2. Those predictions are fed into NN1 to get Vola-Surface again. The results are shown below.

# In[ ]:

"""RESCALING"""



prediction_trafo = prediction.reshape((Ntest,Nparameters,1,1))
forecast = NN1.predict(prediction_trafo).reshape(Ntest,Nmaturities,Nstrikes)
y_true_test = y_test_trafo2.reshape(Ntest,Nmaturities,Nstrikes)
# Example Plots
X = strikes
Y = maturities
X, Y = np.meshgrid(X, Y)
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import random
sample_idx = random.randint(0,len(y_test))

#error plots
mape = np.zeros(forecast.shape)
mse  = np.zeros(forecast.shape)
err_rel_mat  = np.zeros(prediction.shape)
err_mat      = np.zeros(prediction.shape)
for i in range(Ntest):
    mape[i,:,:] =  np.abs((y_true_test[i,:,:]-forecast[i,:,:])/y_true_test[i,:,:])
    mse[i,:,:]  =  np.square((y_true_test[i,:,:]-forecast[i,:,:]))
idx = np.argsort(np.max(mape,axis=tuple([1,2])), axis=None)

#bad_idx = idx[:-200]
bad_idx = idx

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.plot_surface(X, Y, y_true_test[idx[-1],:,:], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, forecast[idx[-1],:,:] , rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Strikes')
ax.set_ylabel('Maturities')
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.plot_surface(X, Y, y_true_test[sample_idx,:,:], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, forecast[sample_idx,:,:] , rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Strikes')
ax.set_ylabel('Maturities')
plt.show()
#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
ax=plt.subplot(2,3,1)
err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(mse[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(mse[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()
plt.savefig('Ultra_NN1NN2_1.png', dpi=300)

# In[ ]:


#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
plt.suptitle('Error with constraint violation', fontsize=16)
ax=plt.subplot(2,3,1)
bad_idx = testing_violation
err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(mse[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(mse[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()
plt.savefig('Ultra_nn1NN2_2.png', dpi=300)

# In[ ]:


#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
plt.suptitle('Error with no constrain violation', fontsize=16)
ax=plt.subplot(2,3,1)
bad_idx = testing_violation2
err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(mse[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(mse[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()

plt.savefig('Ultra_nn1NN2_3.png', dpi=300)

# # 3.1 No parameter constraint

# In[ ]:


NN2a = Sequential() 
NN2a.add(InputLayer(input_shape=(Nmaturities,Nstrikes,1)))
NN2a.add(Conv2D(64,(3, 3),use_bias= True, padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(MaxPooling2D(pool_size=(2, 2)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(ZeroPadding2D(padding=(1,1)))
NN2a.add(Conv2D(64,(2, 2),use_bias= True,padding='valid',strides =(1,1),activation ='tanh'))
NN2a.add(Flatten())
NN2a.add(Dense(5,activation = 'linear',use_bias=True))
NN2a.summary()


NN2a.compile(loss = "MSE", optimizer = "adam",metrics=["MAPE","MSE"])
#NN2.compile(loss =mse_constraint(0.25), optimizer = "adam",metrics=["MAPE", "MSE"])
#NN2.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2),
#        epochs = 50, verbose = True, shuffle=1)
history = NN2a.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2),
    epochs=200, verbose = True, shuffle=1,callbacks=[es])

NN2a.save_weights("Ultra_NN2a.h5")
# In[ ]:


prediction = NN2a.predict(y_test_trafo2)

prediction_invtrafo= np.array([myinverse(x) for x in prediction])

prediction = NN2a.predict(y_test_trafo2)
prediction_std = np.std(prediction,axis=0)
error = np.zeros((Ntest,Nparameters))
for i in range(Ntest):
    error[i,:] =  np.abs((X_test_trafo2[i,:]-prediction[i,:])/X_test_trafo2[i,:])
err1 = np.mean(error,axis = 0)
err2 = np.median(error,axis = 0)
err_std = np.std(error,axis = 0)
idx = np.argsort(error[:,0], axis=None)
good_idx = idx[:-100]
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
_,_,c =constraint_violation(prediction_invtrafo)
_,_,c2 =constraint_violation(X_test)


testing_violation = c>=1
testing_violation2 = (c<1)
plt.figure(figsize=(14,4))
ax=plt.subplot(1,3,1)
plt.boxplot(error)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors")

vio_error = error[testing_violation,:]
vio_error2 = error[testing_violation2,:]
ax=plt.subplot(1,3,2)

plt.boxplot(vio_error)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors parameter violation")
ax=plt.subplot(1,3,3)

plt.boxplot(vio_error2)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors no parameter violation")
plt.show()
plt.savefig('Ultra_NN2a_1.png', dpi=300)

print("error mean in %:",100*err1)
print("error median in %:",100*err2)
print("violation error mean in %:",100*np.mean(vio_error,axis=0))
print("no violation error mean in %:",100*np.mean(vio_error2,axis=0))
print("violation error median in %:",100*np.median(vio_error,axis=0))
print("no violation error median in %:",100*np.median(vio_error2,axis=0))


fig = plt.figure()
plt.scatter(c2,c)
plt.plot(np.arange(0, np.max(c),0.5),np.arange(0, np.max(c),0.5),'-r')
plt.xlabel("True Constraint")
plt.ylabel("Forecasted Constraint")
plt.savefig('Ultra_NN2a_2.png', dpi=300)

plt.figure(figsize=(14,4))
ax=plt.subplot(1,5,1)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
plt.scatter(c2[testing_violation],vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,2)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
plt.scatter(c2[testing_violation],vio_error[:,1],c="b",s=1,marker="x",label="beta con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,3)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
plt.scatter(c2[testing_violation],vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,4)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
plt.scatter(c2[testing_violation],vio_error[:,3],c="b",s=1,marker="x",label="omega con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,5)
plt.yscale("log")
plt.scatter(c2[testing_violation2],vio_error2[:,4],c="r",s=1,marker="x",label="sigma0 no con")
plt.scatter(c2[testing_violation],vio_error[:,4],c="b",s=1,marker="x",label="sigma0 con")
plt.xlabel("True Constraint")
plt.ylabel("Relative Deviation")
plt.legend()
plt.savefig('Ultra_NN2a_5.png', dpi=300)
fig = plt.figure()
plt.scatter(c2,c)
plt.plot(np.arange(0, np.max(c),0.5),np.arange(0, np.max(c),0.5),'-r')
plt.xlabel("True Constraint")
plt.ylabel("Forecasted Constraint")
plt.savefig('Ultra_NN2a_3.png', dpi=300)
plt.figure(figsize=(14,4))
ax=plt.subplot(1,5,1)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,0],c="r",s=1,marker="x",label="alpha no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,0],c="b",s=1,marker="x",label="alpha con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,2)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,1],c="r",s=1,marker="x",label="beta no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,1],c="b",s=1,marker="x",label="beta con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,3)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,2],c="r",s=1,marker="x",label="gamma no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,2],c="b",s=1,marker="x",label="gamma con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,4)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,3],c="r",s=1,marker="x",label="omega no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,3],c="b",s=1,marker="x",label="omega con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
ax=plt.subplot(1,5,5)
plt.yscale("log")
plt.xscale("log")
plt.scatter(abs((c2[testing_violation2]-c[testing_violation2])/c2[testing_violation2]),vio_error2[:,4],c="r",s=1,marker="x",label="sigma0 no con")
plt.scatter(abs((c2[testing_violation]-c[testing_violation])/c2[testing_violation]),vio_error[:,4],c="b",s=1,marker="x",label="sigma0 con")
plt.xlabel("constraint deviation")
plt.ylabel("Relative Deviation")
plt.legend()
plt.savefig('Ultra_NN2a_4.png', dpi=300)

# # 3.2a Testing the performace of the AutoEncoder/Decoder Combination
# We test how the two previously trained NN work together. First, HNG-Vola surfaces are used to predict the underlying parameters with NN2. Those predictions are fed into NN1 to get Vola-Surface again. The results are shown below.

# In[ ]:


prediction_trafo = prediction.reshape((Ntest,Nparameters,1,1))
forecast = NN1.predict(prediction_trafo).reshape(Ntest,Nmaturities,Nstrikes)
y_true_test = y_test_trafo2.reshape(Ntest,Nmaturities,Nstrikes)
# Example Plots
X = strikes
Y = maturities
X, Y = np.meshgrid(X, Y)
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import random
sample_idx = random.randint(0,len(y_test))

#error plots
mape = np.zeros(forecast.shape)
mse  = np.zeros(forecast.shape)
for i in range(Ntest):
    mape[i,:,:] =  np.abs((y_true_test[i,:,:]-forecast[i,:,:])/y_true_test[i,:,:])
    mse[i,:,:]  =  np.square((y_true_test[i,:,:]-forecast[i,:,:]))
idx = np.argsort(np.max(mape,axis=tuple([1,2])), axis=None)

#bad_idx = idx[:-200]
bad_idx = idx
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.plot_surface(X, Y, y_true_test[idx[-1],:,:], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, forecast[idx[-1],:,:] , rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Strikes')
ax.set_ylabel('Maturities')
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.plot_surface(X, Y, y_true_test[sample_idx,:,:], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, forecast[sample_idx,:,:] , rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Strikes')
ax.set_ylabel('Maturities')
plt.show()

#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
ax=plt.subplot(2,3,1)
err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(mse[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(mse[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()
plt.savefig('Ultra_NN1NN2a_1.png', dpi=300)


# In[ ]:


#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
fig.suptitle('Error with no constrain violation', fontsize=16)
ax=plt.subplot(2,3,1)
bad_idx = testing_violation2
err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(mse[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(mse[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()

plt.savefig('Ultra_NN1NN2a_2.png', dpi=300)

# In[ ]:


#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
plt.suptitle('Error with constraint violation', fontsize=16)
ax=plt.subplot(2,3,1)
bad_idx = testing_violation
err1 = 100*np.mean(mape[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(mape[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(mape[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(mse[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(mse[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(mse[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.show()

plt.savefig('Ultra_NN1NN2a_3.png', dpi=300)

# # 5. FFNN as Encoder / Pricing Kernel - Approach of Horvath (improved)

# In[ ]:


X_test_trafo3  = X_test_trafo.reshape((Ntest,5))
X_train_trafo3 = X_train_trafo.reshape((Ntrain,5))
X_val_trafo3   = X_val_trafo.reshape((Nval,5))
#Neural Network
NN3 = Sequential()
NN3.add(InputLayer(input_shape=(Nparameters,)))
NN3.add(Dense(30, activation = 'elu'))
NN3.add(Dense(30, activation = 'elu'))
#NN3.add(Dropout(0.05))
NN3.add(Dense(30, activation = 'relu'))
NN3.add(Dense(Nstrikes*Nmaturities, activation = 'linear', use_bias=True,kernel_constraint = tf.keras.constraints.NonNeg()))
NN3.summary()
"""
#Neural Network Horvath
NN4 = Sequential()
NN4.add(InputLayer(input_shape=(Nparameters,)))
NN4.add(Dense(30, activation = 'elu'))
NN4.add(Dense(30, activation = 'elu'))
NN4.add(Dense(30, activation = 'elu'))
NN4.add(Dense(Nstrikes*Nmaturities, activation = 'linear'))
NN4.summary()
"""               
NN3.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MSE","MAPE"])
#NN3.compile(loss = "mean_squared_error", optimizer = "adam",metrics=["MAPE"])
#NN3.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","mean_squared_error"])
#NN3.compile(loss = 'mean_absolute_percentage_error', optimizer = "adam")
NN3.fit(X_train_trafo3, y_train, batch_size=32, validation_data = (X_val_trafo3, y_val),
        epochs = 200, verbose = True, shuffle=1,callbacks=[es])
#NN4.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MSE"])
#NN4.compile(loss = "mean_squared_error", optimizer = "adam",metrics=["MAPE"])
#NN4.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","mean_squared_error"])
#NN4.compile(loss = 'mean_absolute_percentage_error', optimizer = "adam")
#NN4.fit(X_train_trafo3, y_train, batch_size=32, validation_data = (X_val_trafo3, y_val),
#       epochs = 200, verbose = True, shuffle=1)


# ### 5.1 Results
# __Following results are not correct!!__

# In[ ]:


#error plots
S0=1.
y_test_re       = yinversetransform(y_test)
prediction_list = [yinversetransform(NN3.predict(X_test_trafo3[i].reshape(1,Nparameters))[0]) for i in range(len(X_test_trafo3))]
prediction      = np.asarray(prediction_list)
Ntest           = prediction.shape[0]
err_rel_list = np.abs((y_test_re-prediction)/y_test_re)
err_rel_mat  = err_rel_list.reshape((Ntest,Nmaturities,Nstrikes))
idx = np.argsort(np.max(err_rel_mat,axis=tuple([1,2])), axis=None)
#bad_idx = idx[:-200]
err_list = np.square((y_test_re-prediction))
err_mat  = err_list.reshape((Ntest,Nmaturities,Nstrikes))
bad_idx = idx
#from matplotlib.colors import LogNorm
plt.figure(figsize=(14,4))
ax=plt.subplot(2,3,1)
err1 = 100*np.mean(err_rel_mat[bad_idx,:,:],axis=0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,2)
err2 = 100*np.std(err_rel_mat[bad_idx,:,:],axis = 0)
plt.title("Std relative error",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,3)
err3 = 100*np.max(err_rel_mat[bad_idx,:,:],axis = 0)
plt.title("Maximum relative error",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,4)
err1 = np.sqrt(np.mean(err_mat[bad_idx,:,:],axis=0))
plt.title("RMSE",fontsize=15,y=1.04)
plt.imshow(err1)#,norm=LogNorm(vmin=err1.min(), vmax=err1.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,5)
err2 = np.std(err_mat[bad_idx,:,:],axis = 0)
plt.title("Std MSE",fontsize=15,y=1.04)
plt.imshow(err2)#,norm=LogNorm(vmin=err2.min(), vmax=err2.max()))
plt.colorbar(format=mtick.PercentFormatter())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)
ax=plt.subplot(2,3,6)
err3 = np.max(err_mat[bad_idx,:,:],axis = 0)
plt.title("Maximum MSE",fontsize=15,y=1.04)
plt.imshow(err3)#,norm=LogNorm(vmin=err3.min(), vmax=err3.max()))
plt.colorbar(format=mtick.PercentFormatter())
ax.set_xticks(np.linspace(0,Nstrikes-1,Nstrikes))
ax.set_xticklabels(strikes)
ax.set_yticks(np.linspace(0,Nmaturities-1,Nmaturities))
ax.set_yticklabels(maturities)
plt.xlabel("Strike",fontsize=15,labelpad=5)
plt.ylabel("Maturity",fontsize=15,labelpad=5)

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
plt.savefig('HNG_NN_old_Errors.png', dpi=300)
plt.show()



#==============================================================================
#surface
import random
test_sample = random.randint(0,len(y_test))
test_sample = idx[-1]
#test_sample = idx[0]

y_test_sample = y_test_re[test_sample,:]
y_predict_sample = prediction_list[test_sample]
y_test_sample_p = np.reshape(y_test_sample, (Nmaturities, Nstrikes))
y_predict_sample_p = np.reshape(y_predict_sample, (Nmaturities, Nstrikes))
diff = y_test_sample_p-y_predict_sample_p 
rel_diff = np.abs(y_test_sample_p-y_predict_sample_p)/(y_test_sample_p)
    
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = strikes
Y = maturities
X, Y = np.meshgrid(X, Y)


#ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.plot_surface(X, Y, y_test_sample_p, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.plot_surface(X, Y, y_predict_sample_p , rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
#ax.plot_surface(X, Y, rel_diff, rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none')
ax.set_xlabel('Strikes')
ax.set_ylabel('Maturities')
#ax.set_zlabel('rel. err');
plt.show()


#==============================================================================
#smile
sample_ind = 13
X_sample = X_test_trafo3[sample_ind]
y_sample = y_test[sample_ind]
#print(scale.inverse_transform(y_sample))

prediction=yinversetransform(NN3.predict(X_sample.reshape(1,Nparameters))[0])
plt.figure(figsize=(14,12))
for i in range(Nmaturities):
    plt.subplot(4,4,i+1)
    
    plt.plot(np.log(strikes/S0),y_sample[i*Nstrikes:(i+1)*Nstrikes],'b',label="Input data")
    plt.plot(np.log(strikes/S0),prediction[i*Nstrikes:(i+1)*Nstrikes],'--r',label=" NN Approx")
    #plt.ylim(0.22, 0.26)
    
    plt.title("Maturity=%1.2f "%maturities[i])
    plt.xlabel("log-moneyness")
    plt.ylabel("Implied vol")
    
    plt.legend()
plt.tight_layout()
plt.savefig('HNG_old_smile.png', dpi=300)
plt.show()


# In[ ]:





# In[ ]:




