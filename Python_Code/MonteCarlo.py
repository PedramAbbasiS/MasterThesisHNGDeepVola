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
from help_fun import HNG_MC  

#model parameters 
# Szenario Analyse
#=============================================================================
dt =1
sz_alpha = [0.01,0.02]
sz_gamma = [0.2,0.3]
sz_beta = [0.2,0.5]
sz_lambda = [-0.5,1.3]
sz_omega =[0.1,0.2]
sz_maturity = [10,20]
sz_S0  = [1]
sz_rate = [-0.02/252,-0.1/252,0,0.1/252,0.2/252,0.05/252]
szenario_params_calls ={}
Moneyness = np.array([0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1])
 
for alpha in sz_alpha:
    for beta in sz_beta:
        for gamma in sz_gamma:       
            for d_lambda in sz_lambda:
                gamma_star = gamma+d_lambda+0.5                       
                for omega in sz_omega:                         
                    sigma2 =  omega/(1-alpha*gamma**2-beta) #unconditional variance as garch initialisation
                    for T in sz_maturity: #eventuell unnötig langer pfad modellieren und kürzen?!                            
                        for S_0 in sz_S0:
                            K = Moneyness*S_0
                            for rate in sz_rate:        
                                p_call = HNG_MC(alpha,beta,gamma,omega,d_lambda,S_0,K,rate,T,dt,PutCall = 1)
                                szenario_params_calls[(alpha,beta,gamma,omega,d_lambda,T,S_0,rate)] =  p_call
                                
#=============================================================================

dt = 1                                          #Zeitschritt                        
alpha = 0.01    
beta = 0.2
d_lambda = 1.4                                  #real world
d_lambda_star = -0.5                            #risk-free 
gamma = 0.2                                     #real world
gamma_star = gamma+d_lambda+0.5                 #risk-free  
omega = 0.1                                     #intercept       
PutCall = 1
S=30
r=0.03/252
K=np.array([17,40])
T = 20
sigma2 = (omega+alpha)/(1-alpha*gamma**2-beta)          #unconditional variance VGARCH
V = sigma2

price_rn= hng.HNC(alpha, beta, gamma_star, omega, d_lambda_star, V, S, K, r, T, PutCall) #V= inital sigma2
price_MC_rn = HNG_MC(alpha,beta,gamma,omega,d_lambda,S,K,r,T,dt,PutCall,Variance_specs = "unconditional",output="price")
vola = HNG_MC(alpha, beta, gamma, omega, d_lambda, S, K, r, T, dt, PutCall , num_path = int(1e6), risk_neutral = True, Variance_specs = "unconditional",output="bsvola")
  
print(price_rn,price_MC_rn)
print((price_rn-price_MC_rn)/price_rn*100)
#price_p = hng.HNC(alpha, beta, gamma, omega, d_lambda, V, S, K, r, T, PutCall)
#price_MC_p = HNG_MC(alpha,beta,gamma,omega,d_lambda,S,K,r,T,PutCall,risk_neutral =False)

