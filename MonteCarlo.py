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
from help_fun import *  
"""    
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
"""   
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

dt = 1                                          #Zeitschritt                        
alpha = 0.01    
beta = 0.2
gamma = 0.2                                     #real world
gamma_star = gamma+d_lambda+0.5                 #risk-free  
omega = 0.1                                     #intercept
d_lambda = 1.4                                  #real world
d_lambda_star = -0.5                            #risk-free        
PutCall = 1
S=30
r=0.03/252
K=17
T = 20
sigma2 = omega/(1-alpha*gamma**2-beta)          #unconditional variance VGARCH
V = sigma2

price_rn= hng.HNC(alpha, beta, gamma_star, omega, d_lambda_star, V, S, K, r, T, PutCall) #V= inital sigma2
price_MC_rn = HNG_MC(alpha,beta,gamma,omega,d_lambda,S,K,r,T,dt,PutCall)
print(price_rn,price_MC_rn)
print((price_rn-price_MC_rn)/price_rn*100)
#price_p = hng.HNC(alpha, beta, gamma, omega, d_lambda, V, S, K, r, T, PutCall)
#price_MC_p = HNG_MC(alpha,beta,gamma,omega,d_lambda,S,K,r,T,PutCall,risk_neutral =False)
