### additional functions for main file
import numpy as np
from tensorflow.keras import backend as K
from config import Nparameters,diff,bound_sum,ub,lb,Ntest,Nstrikes,strikes,Nmaturities,maturities
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import matplotlib.lines as mlines
#import matplotlib.transforms as mtransforms
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D  
#from matplotlib import cm


### scaling tools
def ytransform(y_train,y_val,y_test):
    #return [scale.transform(y_train),scale.transform(y_val), 
    #        scale.transform(y_test)]
    return [y_train,y_val,y_test]

def yinversetransform(y):
    return y
    #return scale.inverse_transform(y)
    
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


### custom errors
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


### constraints
def constraint_violation(x):
    return np.sum(x[:,0]*x[:,2]**2+x[:,1]>=1)/x.shape[0],x[:,0]*x[:,2]**2+x[:,1]>=1,x[:,0]*x[:,2]**2+x[:,1]

### error plot
def pricing_plotter(prediction,y_test):     
    err_rel_mat  = np.zeros(prediction.shape)
    err_mat      = np.zeros(prediction.shape)
    for i in range(Ntest):
        err_rel_mat[i,:,:] =  np.abs((y_test[i,:,:]-prediction[i,:,:])/y_test[i,:,:])
        err_mat[i,:,:] =  np.square((y_test[i,:,:]-prediction[i,:,:]))
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
    return err_rel_mat,err_mat,idx,bad_idx