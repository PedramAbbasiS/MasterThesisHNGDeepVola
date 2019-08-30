# Neuronal Network 1 for learning the implied vola 
# source: https://github.com/amuguruza/NN-StochVol-Calibrations/blob/master/1Factor/Flat%20Forward%20Variance/NN1Factor.ipynb 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import py_vollib.black_scholes.implied_volatility as vol
import time
import scipy

###matlab
import scipy.io
#mat = scipy.io.loadmat('data_v2_2000_new.mat')
#data = mat['data']
mat = scipy.io.loadmat('data_vola_mle.mat')
data = mat['data_vola']
#######

#data = np.load('data_test_small_new1.npy')
Nparameters = 5
#maturities = np.array([30, 60, 90, 120, 150, 180, 210, 240])
maturities = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
strikes = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
Nstrikes = len(strikes)   
Nmaturities = len(maturities)   
xx=data[:,:Nparameters]
#yy=data[:,Nparameters+1:]
yy=data[:,Nparameters:]

#####
# split into train and test sample
X_train, X_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15, random_state=42)

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
#NN1.add(Dropout(0.05))
NN1.add(Dense(30, activation = 'elu'))
#NN1.add(Dense(30, activation = 'elu'))
NN1.add(Dense(Nstrikes*Nmaturities, activation = 'linear'))
NN1.summary()

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def root_relative_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)/y_true))    
        
NN1.compile(loss = root_mean_squared_error, optimizer = "adam")
NN1.fit(X_train_trafo, y_train_trafo, batch_size=32, validation_data = (X_val_trafo, y_val_trafo),
        epochs = 200, verbose = True, shuffle=1)
#NN1.save_weights('NN_HNGarch_weights.h5')


#test = yinversetransform(NN1.predict(X_test_trafo))
#y_test_trans = yinversetransform(y_test_trafo)
#error = np.abs((test - y_test_trans))/np.abs(y_test_trans)
#error1 = np.zeros((len(error[0,:])))
#for i in range(len(error[0,:])):
#    error1[i] = np.mean(error[:,i])
    
#==============================================================================
#error plots
S0=1.
y_test_re = yinversetransform(y_test_trafo)
prediction=[yinversetransform(NN1.predict(X_test_trafo[i].reshape(1,Nparameters))[0]) for i in range(len(X_test_trafo))]
plt.figure(1,figsize=(14,4))
ax=plt.subplot(1,3,1)
err = np.mean(100*np.abs((y_test_re-prediction)/y_test_re),axis = 0)
plt.title("Average relative error",fontsize=15,y=1.04)
plt.imshow(err.reshape(Nmaturities,Nstrikes))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
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
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
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
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.tight_layout()
#plt.savefig('HNG_NNErrors.png', dpi=300)
plt.show()



#==============================================================================
#vola surface
import random
test_sample = random.randint(0,len(y_test))
y_test_sample = y_test_re[test_sample,:]
y_predict_sample = prediction[test_sample]

y_test_sample_p = np.reshape(y_test_sample, (Nmaturities, Nstrikes))
y_predict_sample_p = np.reshape(y_predict_sample, (Nmaturities, Nstrikes))
diff = y_test_sample_p-y_predict_sample_p 

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
ax.set_xlabel('Strikes')
ax.set_ylabel('Maturities')
ax.set_zlabel('Volatility');
plt.show()


#==============================================================================
#smile
sample_ind = 12
X_sample = X_test_trafo[sample_ind]
y_sample = y_test[sample_ind]
#print(scale.inverse_transform(y_sample))

prediction=yinversetransform(NN1.predict(X_sample.reshape(1,Nparameters))[0])
plt.figure(1,figsize=(14,12))
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
#plt.savefig('HNG_smile.png', dpi=300)
plt.show()

#==============================================================================
# gradient methods for optimization with Levenberg-Marquardt
NNParameters=[]
for i in range(0,len(NN1.layers)):
    NNParameters.append(NN1.layers[i].get_weights())

NumLayers=3
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

Approx=[]
Timing=[]
#sample_ind = 500
#X_sample = X_test_trafo[sample_ind]
#y_sample = y_test[sample_ind]
solutions=np.zeros([1,Nparameters])
#times=np.zeros(1)
init=np.zeros(Nparameters)
n = 500
for i in range(n):
    disp=str(i+1)+"/5000"
    print (disp,end="\r")
    #L-BFGS-B
    #start= time.clock()
    #I=scipy.optimize.minimize(CostFunc,x0=init,args=i,method='L-BFGS-B',jac=Jacobian,tol=1E-10,options={"maxiter":5000})
    #end= time.clock()
    #solutions=myinverse(I.x)
    #times=end-start
    #Levenberg-Marquardt
    start= time.clock()
    I=scipy.optimize.least_squares(CostFuncLS, init, JacobianLS, args=(i,), gtol=1E-10)
    end= time.clock()
    solutions=myinverse(I.x)
    times=end-start
    
    Approx.append(np.copy(solutions))
    Timing.append(np.copy(times))
LMParameters=[Approx[i] for i in range(len(Approx))]
#np.savetxt("NNParametersHNG.txt",LMParameters)

#==============================================================================
#Calibration Errors with Levenberg-Marquardt¶
titles=["$\\alpha$","$\\beta$","$\\gamma$","$\\omega$", "$\\sigma$"]
average=np.zeros([Nparameters,n])
fig=plt.figure(figsize=(12,8))
for u in range(Nparameters):
    ax=plt.subplot(2,3,u+1)
    for i in range(n):
        
        X=X_test[i][u]
        plt.plot(X,100*np.abs(LMParameters[i][u]-X)/np.abs(X),'b*')
        average[u,i]=np.abs(LMParameters[i][u]-X)/np.abs(X)
    plt.title(titles[u],fontsize=20)
    plt.ylabel('relative Error',fontsize=15)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter() )
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.text(0.5, 0.8, 'Average: %1.2f%%\n Median:   %1.2f%% '%(np.mean(100*average[u,:]),
                                                                np.quantile(100*average[u,:],0.5)), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=15)

    print("average= ",np.mean(average[u,:]))
plt.tight_layout()
#plt.savefig('HNG_ParameterRelativeErrors.png', dpi=300)
plt.show()



#==============================================================================
for u in range(Nparameters):
    for i in range(n):
        Y = y_test[i,:]
        Y_pred = NN1.predict(x.reshape(1,Nparameters))[0]
        X=X_test[i][u]
        plt.plot(X,100*np.abs(LMParameters[i][u]-X)/np.abs(X),'b*')

