#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 08:15:08 2019

@author: Lukas
"""

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
Maturity = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
#K = np.array([0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1])
K = np.array([0.8, 0.84, 0.89, 0.93, 0.98, 1.02, 1.07, 1.11, 1.16, 1.2])
S = 1
r = 0
j = 0
l = 0
d_lambda = 0
Nsim = int(1e2)         #number of samples
Nmaturities = len(Maturity)
Nstrikes = len(K)
a = 0


for i in range(Nsim):
    price = np.zeros((len(Maturity),len(K)))
    a = a + 1
    ###########################################################################
    #only to watch computational status
    j = j+1
    if (j == 50):
        l = l + 1
        print(l*50)
        j = 0 
    ###########################################################################    
    #while ((price < 1e-30).any()) or ((price > .33).any()):  #to avoid numerical problems and too extreme scenarios
    c = 0
    alpha = 1
    beta = .589
    gamma_star = 463.3
    while (beta+alpha*gamma_star**2 > 1):           #stationary condition
        c = c +1 
        print("simulation: ", a, "stationary: ", c)
        #alpha = np.random.uniform(low=9e-7, high=2e-6)
        #beta = np.random.uniform(low=.45, high=.65)  
        #gamma_star = np.random.uniform(low=400, high=550)
        alpha = np.random.uniform(low=-3e-6, high=1.59e-6)
        #beta = np.random.uniform(low=.45, high=.65) 
    #omega = np.random.uniform(low=2e-6, high=8e-6) 
    omega = np.random.uniform(low=7.55e-6, high=3.45e-4) 
    #d_lambda = np.random.uniform(low=.05, high=.6)
    for t in range(len(Maturity)):
        for k in range(len(K)):
            price[t,k] = hng.HNC(alpha, beta, gamma_star, omega, -.5, 0.04/252, 
                S, K[k], r, Maturity[t], PutCall=1)
    #price = HNG_MC_simul(alpha, beta, gamma_star, omega, d_lambda, S, K,
    #                     r, Maturity, dt, output=1, risk_neutral = False)
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
density = gaussian_kde(iv_re)
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.show()
