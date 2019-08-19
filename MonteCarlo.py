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
from scipy.optimize import minimize
import hngoption as hng

def HNG_MC(alpha,beta,gamma,omega,lam,S_0,K,rate,T,PutCall,num_path = 1000000,risk_neutral = True):
    sigma2 =  omega/(1-alpha*gamma**2-beta)
    r = np.exp(rate*dt)-1                             
    lnS = np.zeros((num_path,T+1))
    h = np.zeros((num_path,T+1))
    lnS[:,0] = np.log(S_0)*np.ones((num_path))
    h[:,0] = sigma2*np.ones((num_path)) #initial wert
    z = np.random.normal(size=(num_path,T+1))
    gamma_star = gamma+lam+0.5
    for t in np.arange(dt,T+dt,dt):
        if risk_neutral:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma_star*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r-0.5*h[:,t]+np.sqrt(h[:,t])*z[:,t]
        else:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r+lam*h[:,t]+np.sqrt(h[:,t])*z[:,t]
    S_T = np.exp(lnS[:,-1])
    if PutCall:
        return np.exp(-rate*T)*np.mean(np.maximum(S_T-K,0))
    else:
        return np.exp(-rate*T)*np.mean(np.maximum(K-S_T,0))    
    
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
# =============================================================================
#     
# for alpha in sz_alpha:
#         for beta in sz_beta:
#             for gamma in sz_gamma:
#                 for lam in sz_lambda:
#                     for omega in sz_omega:
#                         sigma2 =  omega/(1-alpha*gamma**2-beta) #unconditional variance as garch initialisation
#                         #underlyings
#                         for T in sz_maturity:#eventuell unnötig langer pfad modellieren und kürzen?!
#                             for S_0 in sz_S0:#eventuell unnötig (S/K nur relevant? in realen daten? dunno)
#                                 for rate in sz_rate:
#                                     param_option_dict[(alpha,beta,gamma,omega,lam,T,S_0,rate,"c")] =  np.mean(np.maximum(S_T[:,np.newaxis] - K,np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
#                                     param_option_dict[(alpha,beta,gamma,omega,lam,T,S_0,rate,"p")] =  np.mean(np.maximum(K-S_T[:,np.newaxis],np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
#                      
# =============================================================================
                        

                          

alpha = 0.01
beta = 0.2
gamma = 0.2
omega = 0.1
d_lambda = -0.5
PutCall = 1
S=30
r=0.03/252
K=17
T = 20
PutCall = 1
sigma2 = omega/(1-alpha*gamma**2-beta)
V = omega + beta*sigma2+alpha*(-r-d_lambda*sigma2-gamma*sigma2)**2/sigma2
g_star = gamma+d_lambda+0.5

price_rn = hng.HNC(alpha, beta, g_star, omega, -0.5, V, S, K, r, T, PutCall) #V mit adjustment
price_rn= hng.HNC(alpha, beta, g_star, omega, -0.5, sigma2, S, K, r, T, PutCall) #V= inital sigma2
price_MC_rn = HNG_MC(alpha,beta,gamma,omega,d_lambda,S,K,r,T,PutCall)
print(price_rn,price_MC_rn)
print((price_rn-price_MC_rn)/price_rn*100)
#price_p = hng.HNC(alpha, beta, gamma, omega, d_lambda, V, S, K, r, T, PutCall)
#price_idea = hng.HNC(alpha, beta, g_star, omega, -0.5, sigma2, S, K, r, T, PutCall)
#price_MC_p = HNG_MC(alpha,beta,gamma,omega,d_lambda,S,K,r,T,PutCall,risk_neutral =False)
