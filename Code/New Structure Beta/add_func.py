### additional functions for main file
import numpy as np
from tensorflow.keras import backend as K
from config import Nparameters,r,diff,bound_sum,ub,lb,Ntest,Nstrikes,strikes,Nmaturities,maturities
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from py_vollib.black_scholes.implied_volatility import implied_volatility as bsimpvola
import os as os
from multiprocessing import Pool

#import matplotlib.lines as mlines
#import matplotlib.transforms as mtransforms
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D  
#from matplotlib import cm
import cmath
import math

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


### Heston Nandi Pricer
"""
Heston Nandi GARCH Option Pricing Model (2000) 
Based on the code of Dustin Zacharias, MIT, 2017
Code Available under https://github.com/SW71X/hngoption2
"""
# Trapezoidal Rule passing two vectors
def trapz(X, Y):
    n = len(X)
    sum = 0.0
    for i in range(1, n):
        sum += 0.5 * (X[i] - X[i - 1]) * (Y[i - 1] + Y[i])
    return sum

# HNC_f returns the real part of the Heston & Nandi integral with Q parameers
def HNC_f_Q(complex_phi, d_alpha, d_beta, d_gamma_star, d_omega, d_V, d_S, d_K, d_r, i_T, i_FuncNum):
    A = [x for x in range(i_T + 1)]
    B = [x for x in range(i_T + 1)]
    complex_zero = complex(0.0, 0.0)
    complex_one = complex(1.0, 0.0)
    #complex_i = complex(0.0, 1.0)
    A[i_T] = complex_zero
    B[i_T] = complex_zero
    for t in range(i_T - 1, -1, -1):
        if i_FuncNum == 1:
            A[t] = A[t + 1] + (complex_phi + complex_one) * d_r + B[t + 1] * d_omega \
                   - 0.5 * cmath.log(1.0 - 2.0 * d_alpha * B[t + 1])
            B[t] = - 0.5 * (complex_phi+complex_one)+ d_beta * B[t + 1] \
                   + (0.5 * (complex_phi+complex_one) ** 2-2*d_alpha*(d_gamma_star)*B[t+1]*(complex_phi+complex_one)\
                      +d_alpha*(d_gamma_star)**2*B[t+1] )/ (1.0 - 2.0 * d_alpha * B[t + 1])
        else:
            A[t] = A[t + 1] + (complex_phi) * d_r + B[t + 1] * d_omega \
                   - 0.5 * cmath.log(1.0 - 2.0 * d_alpha * B[t + 1])
            B[t] = - 0.5 * complex_phi + d_beta * B[t + 1] \
                   + (0.5 * (complex_phi) **2 - 2*d_alpha*(d_gamma_star)*B[t+1]*complex_phi+d_alpha*(d_gamma_star)**2*B[t+1] )\
                   / (1.0 - 2.0 * d_alpha * B[t + 1])
    if i_FuncNum == 1:
        z = (d_K ** (-complex_phi)) * (d_S ** (complex_phi + complex_one)) \
            * cmath.exp(A[0] + B[0] * d_V) / complex_phi
        return z.real
    else:
        z = (d_K ** (-complex_phi)) * (d_S ** (complex_phi)) * cmath.exp(A[0] + B[0] * d_V) / complex_phi
        return z.real
    
# Returns the Heston and Nandi option price under Q parameters
def HNC_Q(alpha, beta, gamma_star, omega, V, S, K, r, T, PutCall):
    const_pi = 4.0 * math.atan(1.0)
    High = 1000
    Increment = 0.05
    NumPoints = int(High / Increment)
    X, Y1, Y2 = [], [], []
    i = complex(0.0, 1.0)
    phi = complex(0.0, 0.0)
    for j in range(0, NumPoints):
        if j == 0:
            X.append(0.0000001)
        else:
            X.append(j * Increment)
        phi = X[j] * i
        Y1.append(HNC_f_Q(phi, alpha, beta, gamma_star, omega, V, S, K, r, T, 1))
        Y2.append(HNC_f_Q(phi, alpha, beta, gamma_star, omega, V, S, K, r, T, 2))

    int1 = trapz(X, Y1)
    int2 = trapz(X, Y2)
    Call = S / 2 + math.exp(-r * T) * int1 / const_pi - K * math.exp(-r * T) * (0.5 + int2 / const_pi)
    Put = Call + K * math.exp(-r * T) - S
    if PutCall == 1:
        return Call
    else:
        return Put
    return 

def opti_fun_data(prediction):
    def opti_fun(omega,alpha,beta,gamma_star,h0):
        def error_fun(n):
            err = 0
            i=0
            for k in strikes:
                j=0
                for T in maturities:
                    err += ((prediction(n,i,j)-bsimpvola(HNC_Q(alpha, beta, gamma_star, omega, h0, 1, k, r, T, 1),1,k,T,r,'c'))\
                                    /prediction(n,i,j))**2
                    j+=1
                i+=1
            return err/(Ntest*Nmaturities*Nstrikes)
        try:
            pool = Pool(os.cpu_count()) # on 8 processors
            error = np.sum(pool.map(error_fun, range(Ntest)))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
        return error
    return opti_fun