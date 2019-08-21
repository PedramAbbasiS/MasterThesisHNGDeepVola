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


#model parameters 
# Szenario Analyse
#=============================================================================
szenario_data =[]
szenario_data_closed = []
dt =1
Maturity = np.array([20, 30, 80, 180, 250])
K = np.array([0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1])
S = 1
r = 0
j = 0
l = 0
d_lambda = 0
vola_closed = np.zeros((len(Maturity),len(K)))

for i in range(int(50e1)):
    print(i)
    alpha = 1
    beta = 1
    gamma_star = 1
    j = j+1
    if (j == 1000):
        l = l + 1
        print(l*1000)
        j = 0 
    while (beta+alpha*gamma_star**2 > 1):
        alpha = np.random.uniform(low=1e-7, high=1e-4)
        beta = np.random.uniform(low=.3, high=1)  
        gamma_star = np.random.uniform(low=50, high=350)
    omega = np.random.uniform(low=1e-5, high=1e-12)  
    vola = HNG_MC_simul(alpha, beta, gamma_star, omega, d_lambda, S, K, r, Maturity, dt, output=1)
    V = (omega+alpha)/(1-alpha*gamma_star**2-beta) 
    for m in range(len(Maturity)):
        for k in range(len(K)):
            vola_closed[m,k] = hng.HNC(alpha, beta, gamma_star, omega, -.5, V, S, K[k], r, Maturity[m], PutCall=1)
    MC_diff = np.mean(np.abs((vola - vola_closed)/vola_closed))       
    szenario_data.append(np.concatenate((np.asarray([alpha,beta,gamma_star,omega, 
                                                     beta+alpha*gamma_star**2, MC_diff]).reshape((1,6)), vola.reshape((1,vola.shape[0]*vola.shape[1]))),axis=1))   
                               
#szenario_data_1 = np.asarray(szenario_data).reshape((int(4e4),40)) 
szenario_data_2 = np.asarray(szenario_data).reshape((int(50e1),41))
mean_diff = np.mean(szenario_data_2[:,5])
max_diff = np.max(szenario_data_2[:,5])
#np.save('data', szenario_data_1)


                
#szenario_vola_calls[(alpha,beta,gamma_star,omega)] =  vola.reshape((1,vola.shape[0]*vola.shape[1]))
                
# #===========================================================================

               
               
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
