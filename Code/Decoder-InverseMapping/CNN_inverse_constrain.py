"""
Created on Thu Nov 28 13:27:14 2019

@author: Henrik Brautmeier

CNN for decoding!
"""

# Neuronal Network 1 for learning the implied vola 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential,Model
from keras.layers import InputLayer,Dense,Flatten, Conv2D, Dropout, Input,ZeroPadding2D,MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tensorflow as tf
#import py_vollib.black_scholes.implied_volatility as vol
#import time
import scipy
import scipy.io
#alpha beta gamma omega sigma0
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
scale = StandardScaler()
y_train_transform = scale.fit_transform(y_train)
y_test_transform = scale.transform(y_test)
y_val_transform = scale.transform(y_val)
def ytransform(y_train,y_val,y_test):
    return [scale.transform(y_train),scale.transform(y_val), 
            scale.transform(y_test)]
    return [y_train,y_val,y_test]
def yinversetransform(y):
    return y
    return scale.inverse_transform(y)
[y_train_trafo, y_val_trafo, y_test_trafo]=ytransform(y_train, y_val, y_test)
y_train_trafo = np.asarray([y_train[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntrain)])
y_val_trafo =  np.asarray([y_val[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Nval)])
y_test_trafo =  np.asarray([y_test[i,:].reshape((1,Nmaturities,Nstrikes)) for i in range(Ntest)])
ub=np.amax(xx, axis=0)
lb=np.amin(xx, axis=0)
diff = ub-lb
bound_sum =ub+lb
def myscale(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=(2*x[i] - (ub[i] + lb[i])) / (ub[i] - lb[i])
    return res
def myinverse(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=x[i]*(ub[i] - lb[i]) *0.5 + (ub[i] + lb[i])*0.5
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
            return tf.keras.losses.MSE( y_true, y_pred) +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint
"""best_loss = 999
params = np.asarray([0,0.25])
saver = np.zeros(params.shape)
count = 0
for param in params:
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
    #NN2.summary()


    #NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
    NN2.compile(loss =mse_constraint(param), optimizer = "adam",metrics=["MAPE", "MSE"])
    #NN2.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2),
    #        epochs = 50, verbose = True, shuffle=1)
    history = NN2.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2),
        epochs=15, verbose = True, shuffle=1)
    saver[count] = history.history["mean_absolute_percentage_error"][-1]
    if history.history["mean_absolute_percentage_error"][-1] <best_loss :
        best_loss = history.history["mean_absolute_percentage_error"][-1]
        best_history = history
        best_param = param
        NN2.save('best_model.h5')
    count += 1

fig = plt.figure()
plt.scatter(params,saver)
# Recreate the exact same model purely from the file
best_model= keras.models.load_model('best_model.h5')
"""
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
#NN2.summary()


#NN2.compile(loss = root_relative_mean_squared_error, optimizer = "adam",metrics=["MAPE","MSE"])
NN2.compile(loss =mse_constraint(param=0.25), optimizer = "adam",metrics=["MAPE", "MSE"])
#NN2.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2),
#        epochs = 50, verbose = True, shuffle=1)
history = NN2.fit(y_train_trafo2,X_train_trafo2, batch_size=50, validation_data = (y_val_trafo2,X_val_trafo2),
    epochs=35, verbose = True, shuffle=1)

def constraint_violation(x):
    return np.sum(x[:,0]*x[:,2]**2+x[:,1]>=1)/x.shape[0],x[:,0]*x[:,2]**2+x[:,1]>=1,x[:,0]*x[:,2]**2+x[:,1]
prediction = NN2.predict(y_test_trafo2)

prediction_invtrafo= np.array([myinverse(x) for x in prediction])

prediction_std = np.std(prediction,axis=0)
error = np.zeros((Ntest,Nparameters))
for i in range(Ntest):
    error[i,:] =  np.abs((X_test_trafo2[i,:]-prediction[i,:])/X_test_trafo2[i,:])
err1 = np.mean(error,axis = 0)
err2 = np.median(error,axis = 0)
err_std = np.std(error,axis = 0)
idx = np.argsort(error[:,0], axis=None)
good_idx = idx[:-100]

plt.figure(figsize=(14,4))
ax=plt.subplot(1,3,1)
plt.boxplot(error)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors")
plt.show()
print("error mean in %:",100*err1)
print("error median in %:",100*err2)
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
ax=plt.subplot(1,3,2)

plt.boxplot(vio_error)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors parameter violation")
plt.show()
ax=plt.subplot(1,3,3)

plt.boxplot(vio_error2)
plt.yscale("log")
plt.xticks([1, 2, 3,4,5], ['w','a','b','g*','h0'])
plt.ylabel("Errors no parameter violation")
plt.show()
print("violation error mean in %:",100*np.mean(vio_error,axis=0))
print("no violation error mean in %:",100*np.mean(vio_error2,axis=0))
print("violation error median in %:",100*np.median(vio_error,axis=0))
print("no violation error median in %:",100*np.median(vio_error2,axis=0))


fig = plt.figure()
plt.scatter(c2,c)
plt.plot(np.arange(0, np.max(c),0.5),np.arange(0, np.max(c),0.5),'-r')
plt.xlabel("True Constraint")
plt.ylabel("Forecasted Constraint")


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
fig = plt.figure()
plt.scatter(c2,c)
plt.plot(np.arange(0, np.max(c),0.5),np.arange(0, np.max(c),0.5),'-r')
plt.xlabel("True Constraint")
plt.ylabel("Forecasted Constraint")

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
