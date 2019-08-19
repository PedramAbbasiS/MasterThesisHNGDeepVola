# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:41:27 2019

@author: Henrik Brautmeier

Option Pricing in Maschine Learning
"""
# preambel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from statsmodels.tsa.ar_model import AR
from arch.univariate import ConstantMean, GARCH,ARX
from statsmodels.api import qqplot
from statsmodels.tsa.arima_model import ARIMA
import scipy.stats as stats
from scipy.optimize import minimize
import csv
import os
import keras

#Monte Carlo
""" 
Parameters and boundaries:
Maturity T in days  10-252
Starting underlying S_0 in  
r.

"""
dt = 1
T = 10
t = np.arange(0,T+1,dt)
S_0 = np.arange(50,160,10)
r_0 = np.arange(-0.05,0.06,0.01)
lam = 0.5
omega = 0.1
alpha = 0.1
beta =0.1
gamma =-0.1
r_dict= {}
for rate in [-0.05,-0.04,-0.03,-0.02,-0.01,0,0.02,0.03,0.04,0.05]:
    for time in t:
        r_dict[(rate,time)]=np.exp(rate*time)
r = np.exp(np.outer(r_0,t))

lnS =np.zeros(r.shape)
lnS[:,0] = np.log(S_0)
h = np.zeros(r.shape)

sigma2 = omega/(1-alpha*gamma**2-beta)
h[:,0] = sigma2*np.ones(r.shape[0])
z = np.random.normal(size=(r.shape))
for s in np.arange(1,T+1,dt):
    lnS[:,s] = lnS[:,s-dt]+np.exp(r_0*dt)+lam*h[:,s]+np.sqrt(h[:,s])*z[:,s]
    h[:,s] = omega+beta*h[:,s-dt]+alpha*(z[:,s-dt]+gamma*np.sqrt(h[:,s-dt]))**2
#
    
    
import hngoption as hng
dt = 1
num_path = 10000
sz_alpha = [0.001,0.002]
sz_gamma = [0.002,0.03]
sz_beta = [0.001,0.02]
sz_lambda = [-0.0005,0.00003]
sz_omega =[0.0001,0.0002]
sz_maturity = [10,20]
sz_S0  = [1,1.2]
sz_rate = [0,0.01]
sz_moneyness =  np.arange(0.9,1.15,0.05)
param_option_dict ={}
#model parameters    
    
for alpha in sz_alpha:
        for beta in sz_beta:
            for gamma in sz_gamma:
                for lam in sz_lambda:
                    for omega in sz_omega:
                        sigma2 =  omega/(1-alpha*gamma**2-beta) #unconditional variance as garch initialisation
                        #underlyings
                        for T in sz_maturity:#eventuell unnötig langer pfad modellieren und kürzen?!
                            for S_0 in sz_S0:#eventuell unnötig (S/K nur relevant? in realen daten? dunno)
                                for rate in sz_rate:
# =============================================================================
#                                     r = np.exp(rate*dt)-1
#                                     lnS = np.zeros((num_path,T+1))
#                                     h = np.zeros((num_path,T+1))
#                                     lnS[:,0] = np.log(S_0)*np.ones((num_path))
#                                     h[:,0] = sigma2*np.ones((num_path)) #initial wert
#                                     z = np.random.normal(size=(num_path,T+1))
#                                     for t in np.arange(dt,T+dt,dt):
#                                         h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]+gamma*np.sqrt(h[:,t-dt]))**2
#                                         lnS[:,t] = lnS[:,t-dt]+r+lam*h[:,t]+np.sqrt(h[:,t])*z[:,t]
#                                         S_T = np.exp(lnS[:,-1])
#                                     K = sz_moneyness*S_0
# =============================================================================
                                  #  param_option_dict[(alpha,beta,gamma,omega,lam,T,S_0,rate,"c")] =  np.mean(np.maximum(S_T[:,np.newaxis] - K,np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
                                  #  param_option_dict[(alpha,beta,gamma,omega,lam,T,S_0,rate,"p")] =  np.mean(np.maximum(K-S_T[:,np.newaxis],np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
                     
                        

def HNG_MC(alpha,beta,gamma,omega,lam,S_0,K,rate,T,PutCall,num_path = 20000000):
    sigma2 =  omega/(1-alpha*gamma**2-beta)
    r = np.exp(rate*dt)-1                             
    lnS = np.zeros((num_path,T+1))
    h = np.zeros((num_path,T+1))
    lnS[:,0] = np.log(S_0)*np.ones((num_path))
    h[:,0] = sigma2*np.ones((num_path)) #initial wert
    z = np.random.normal(size=(num_path,T+1))
    for t in np.arange(dt,T+dt,dt):
        h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]+gamma*np.sqrt(h[:,t-dt]))**2
        lnS[:,t] = lnS[:,t-dt]+r+lam*h[:,t]+np.sqrt(h[:,t])*z[:,t]
        S_T = np.exp(lnS[:,-1])
    if PutCall:
        return np.exp(-rate*T)*np.mean(np.maximum(S_T-K,0))
    else:
            return np.exp(-rate*T)*np.mean(np.maximum(K-S_T,0))
                          

alpha = 0.0001
beta = 0.002
gamma = 0.002
omega = 0.0001
d_lambda = -0.0005
PutCall = 1
S=1
r=0
K=0.9
T = 10
PutCall = 1
sigma2 = omega/(1-alpha*gamma**2-beta)
V = omega + beta*sigma2+alpha*(-r-d_lambda*sigma2-gamma*sigma2)**2/sigma2

g_star = gamma+d_lambda+0.5
price = hng.HNC(alpha, beta, gamma, omega, d_lambda, V, S, K, r, T, PutCall)
price_riskneutral = hng.HNC(alpha, beta, g_star, omega, -0.5, V, S, K, r, T, PutCall)
price_2 = hng.HNC(alpha, beta, g_star, omega, d_lambda, sigma2, S, K, r, T, PutCall)
price_MC = HNG_MC(alpha,beta,gamma,omega,d_lambda,S,K,r,T,PutCall,num_path = 10000)

print(price,price_aprice_2,price_MC)
print(np.abs(price_riskneutral-price_MC)/price)