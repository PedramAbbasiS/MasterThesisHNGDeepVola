# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:41:27 2019

@author: Henrik Brautmeier

Option Pricing in Maschine Learning
"""
# preambel
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import hngoption as hng
from help_fun import HNG_MC,HNG_MC_simul
from tempfile import TemporaryFile
import py_vollib.black_scholes.implied_volatility as vol

#=============================================================================
szenario_data =[]
dt =1
Maturity = np.array([20, 50, 80, 110, 140])
K = np.array([0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1])
S = 1
r = 0
j = 0
l = 0
d_lambda = 0
Nsim = int(4.5e4)         #number of samples
Nmaturities = len(Maturity)
Nstrikes = len(K)
for i in range(Nsim):
    price = np.zeros((2,2))
    ###########################################################################
    #only to watch computational status
    j = j+1
    if (j == 1000):
        l = l + 1
        print(l*1000)
        j = 0 
    ###########################################################################    
    while ((price < 1e-9).any()) or ((price > .5).any()):  #to avoid numerical problems and too extreme scenarios
        alpha = 1
        beta = 1
        gamma_star = 1
        while (beta+alpha*gamma_star**2 > 1):           #stationary condition
            alpha = np.random.uniform(low=5e-7, high=5e-5)
            beta = np.random.uniform(low=.1, high=.6)  
            gamma_star = np.random.uniform(low=300, high=500)
        omega = np.random.uniform(low=1e-6, high=9e-6) 
        d_lambda = np.random.uniform(low=.4, high=2.4) 
        price = HNG_MC_simul(alpha, beta, gamma_star, omega, d_lambda, S, K,
                             r, Maturity, dt, output=1, risk_neutral = False)  
    szenario_data.append(np.concatenate((np.asarray([alpha,beta,gamma_star,omega,d_lambda, 
                                                     beta+alpha*gamma_star**2]).reshape((1,6)), price.reshape((1,price.shape[0]*price.shape[1]))),axis=1))   
                               
szenario_data_2 = np.asarray(szenario_data).reshape((Nsim,6+Nstrikes*Nmaturities))
print("number of nonzeros price-data: ", np.count_nonzero(szenario_data_2))
print("max price: ", np.max(szenario_data_2[:,6:]))
print("min price: ", np.min(szenario_data_2[:,6:]))
print("mean price: ", np.mean(szenario_data_2[:,6:]))
print("median price: ", np.median(szenario_data_2[:,6:]))

flag = 'c'
y_p=szenario_data_2[:,6:]
iv = np.full_like(y_p, 0.0)
stri = np.zeros((Nmaturities, Nstrikes))
for i in range(Nsim):
    price_new = y_p[i,:].reshape((Nmaturities, Nstrikes))
    for m in range(Nmaturities):
        for k in range(Nstrikes):
            try:
                stri[m,k] = vol.implied_volatility(price_new[m,k], S, K[k], Maturity[m]/252, r, flag)
            except:
                pass
    iv[i] =  stri.reshape((1,len(Maturity)*len(K)))  

print("number of nonteros vola-data: ", np.count_nonzero(iv))
print("max price: ", np.max(iv))
print("min price: ", np.min(iv))
print("mean vola: ", np.mean(iv))
print("mean vola: ", np.median(iv))

data_connect = np.concatenate((szenario_data_2[:,:6], iv), axis=1)
np.save('data_IV_4e4_MC_1e4', data_connect)
#mean_diff = np.mean(szenario_data_2[:,5])
#max_diff = np.max(szenario_data_2[:,5])
#np.save('data_shortmat_MC_1e5', szenario_data_2)


  



# old code
#==============================================================================
#==============================================================================               
#szenario_vola_calls[(alpha,beta,gamma_star,omega)] =  vola.reshape((1,vola.shape[0]*vola.shape[1]))
                
# #===========================================================================
#    for m in range(len(Maturity)):
#        for k in range(len(K)):
#            vola_closed[m,k] = hng.HNC(alpha, beta, gamma_star, omega, -.5, V, S, K[k], r, Maturity[m], PutCall=1)
#    MC_diff = np.mean(np.abs((vola - vola_closed)/vola_closed))  
               
               
#  
#  dt = 1                                          #Zeitschritt                        
#  alpha = 0.01    
#  beta = 0.2
#  d_lambda = 1.4                                  #real world
#  d_lambda_star = -0.5                            #risk-free 
#  gamma = 0.2                                     #real world
#  gamma_star = gamma+d_lambda+0.5                 #risk-free  
#  omega = 0.1                                     #intercept       
#  PutCall = 1
#  S=1
#  r=0.03/252
#  K=np.array([17,40,50])/30
#  T = 20
#  sigma2 = (omega+alpha)/(1-alpha*gamma_star**2-beta)          #unconditional variance VGARCH
#  V = sigma2
#  
#  #price_rn= hng.HNC(alpha, beta, gamma_star, omega, d_lambda_star, V, S, K, r, T, PutCall) #V= inital sigma2
#  from help_fun import HNG_MC_simul
#  T=np.array([10,20])
#  price_MC_rn = HNG_MC_simul(alpha,beta,gamma_star,omega,d_lambda,S,K,r,T,dt)
#  vola = HNG_MC_simul(alpha, beta, gamma_star, omega, d_lambda, S, K, r, T, dt, output=0)
#    
#  print(price_rn,price_MC_rn)
#  print((price_rn-price_MC_rn)/price_rn*100)
#  #price_p = hng.HNC(alpha, beta, gamma, omega, d_lambda, V, S, K, r, T, PutCall)
#  #price_MC_p = HNG_MC(alpha,beta,gamma,omega,d_lambda,S,K,r,T,PutCall,risk_neutral =False)
#  
# # =============================================================================
