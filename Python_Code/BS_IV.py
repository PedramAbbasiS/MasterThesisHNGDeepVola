"""
@author: Henrik Brautmeier, Lukas Wuertenberger

Option Pricing in Maschine Learning
"""
import numpy as np
import matplotlib.pyplot as plt
from help_fun import HNG_MC_simul,data_generator
import py_vollib.black_scholes.implied_volatility as vol
#Source: http://vollib.org/documentation/python/0.1.5/index.html 

sz_alpha = [0.02]
sz_gamma = [0.3]
sz_beta = [0.5]
sz_omega =[0.2]
dt = 1/252 # stepwidth in basis of 1year
Maturity = np.array([20,30,50,60,100,252]) #Maturity always in timesteps of dt > integermatrix
szenario_data =[]
K = np.array([0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1])
S = 1
r = 0
form = 0
szenario_data,szenarios = data_generator(sz_alpha,sz_beta,sz_gamma,sz_omega,K,Maturity,dt,r,1,form)
szenario_data_vola, szenarios= data_generator(sz_alpha,sz_beta,sz_gamma,sz_omega,K,Maturity,dt,r,0,form)
flag = 'c'
iv =[]
l=0
s=0
for k in K:
    
    for t in Maturity/252:
        price = szenario_data[1,s,l]
        iv.append(vol.implied_volatility(price, S, k, t, r, flag))
        s+=1
    l+=1    
        
        
#price = vol.black_scholes(flag, S, K, t, r, sigma)
