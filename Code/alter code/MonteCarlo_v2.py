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
#Maturity = np.array([20, 50, 80, 110, 140])
Maturity = np.array([80, 110, 140, 170, 200])
K = np.array([0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1])
S = 1
r = 0
j = 0
l = 0
d_lambda = 0
Nsim = int(2e3)         #number of samples
Nmaturities = len(Maturity)
Nstrikes = len(K)
for i in range(Nsim):
    price = np.zeros((2,2))
    ###########################################################################
    #only to watch computational status
    j = j+1
    if (j == 50):
        l = l + 1
        print(l*50)
        j = 0 
    ###########################################################################    
    while ((price < 1e-30).any()) or ((price > .33).any()):  #to avoid numerical problems and too extreme scenarios
        alpha = 1
        beta = 1
        gamma_star = 1
        while (beta+alpha*gamma_star**2 > 1):           #stationary condition
            alpha = np.random.uniform(low=9e-7, high=2e-6)
            beta = np.random.uniform(low=.45, high=.65)  
            gamma_star = np.random.uniform(low=400, high=550)
        omega = np.random.uniform(low=2e-6, high=8e-6) 
        d_lambda = np.random.uniform(low=.05, high=.6) 
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

print("median alpha: ", np.median(szenario_data_2[:,0]))
print("median beta: ", np.median(szenario_data_2[:,1]))
print("median gamma: ", np.median(szenario_data_2[:,2]))
print("median omega: ", np.median(szenario_data_2[:,3]))
print("median lambda: ", np.median(szenario_data_2[:,4]))
print("median stationary constraint: ", np.median(szenario_data_2[:,5]))

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
            except Exception:
                #pass
                stri[m,k] = np.nan
    iv[i] =  stri.reshape((1,len(Maturity)*len(K)))  
iv1 = iv[(~np.isnan(iv)).all(axis=1)]

print("number of nonteros vola-data: ", np.count_nonzero(iv1))
print("max vola: ", np.max(iv1))
print("min vola: ", np.min(iv1))
print("mean vola: ", np.mean(iv1))
print("median vola: ", np.median(iv1))
print("low volas: ", len(iv1[(iv1<.07).any(axis=1)]))

data_connect = np.concatenate((szenario_data_2[:,:6], iv), axis=1)
data_connect = data_connect[(~np.isnan(data_connect)).all(axis=1)]
np.save('data_test_small_new1', data_connect)
#mean_diff = np.mean(szenario_data_2[:,5])
#max_diff = np.max(szenario_data_2[:,5])
#np.save('data_shortmat_MC_1e5', szenario_data_2)


  

#alpha = 3e-8
#beta = .2
#gamma_star = 500
#omega = 7.6e-6
#d_lambda = 2.0
#price =  HNG_MC_simul(alpha, beta, gamma_star, omega, d_lambda, S, K,
#                             r, Maturity, dt, output=1, risk_neutral = False)
#vola = vol.implied_volatility(np.min(price), S, K[6], 80/252, r, flag)
#print(vola)



###plot
iv_re = iv1.reshape((1,len(iv1)*Nstrikes*Nmaturities))
from matplotlib import pyplot as plt 
from scipy.stats import gaussian_kde
#density = gaussian_kde(iv_re)
density = gaussian_kde(data[:,5])
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.show()



# old code
#data1 = data[(data!=0).all(axis=1)]
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