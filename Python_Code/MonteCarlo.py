# -*- coding: utf-8 -*-
"""
@author: Henrik Brautmeier, Lukas Wuertenberger

Option Pricing in Maschine Learning
"""
# preambel
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import minimize
#import hngoption as hng
from help_fun import HNG_MC_simul,data_generator
import keras

#model parameters 
# Szenario Analyse
#=============================================================================
sz_alpha = [0.01,0.02]
sz_gamma = [0.2,0.3]
sz_beta = [0.2,0.5]
#sz_lambda = [-0.5,1.3]
sz_omega =[0.1,0.2]
dt = 1/252 # stepwidth in basis of 1year
Maturity = np.array([10,20,30,40,50,60,100,252]) #Maturity always in timesteps of dt > integermatrix
#sz_S0  = [1] #normalization
#sz_rate = [-0.02/252,-0.1/252,0,0.1/252,0.2/252,0.05/252]
#szenario_vola_calls ={}
szenario_data =[]
K = np.array([0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1])

r = 0 # yearl rate times dt example 5%*1/252
value = 1
# use form=1 for usual format form = 0 for matrices
form = 0
szenario_data,szenarios = data_generator(sz_alpha,sz_beta,sz_gamma,sz_omega,K,Maturity,dt,r,value,form)

#surface plots================================================================
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
X, Y = np.meshgrid(K,Maturity)
Z = szenario_data[15,:,:]
fig = plt.figure()
ax = fig.gca(projection='3d')
sz_num = 1
surf = ax.plot_surface(X, Y, szenario_data[sz_num,:,:], rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
 










# Testin area ================================================================
#!pip install scikit-image
#!pip install -U efficientnet
#import efficientnet as efn 
#model = efn.EfficientNetB0()
#model.compile(loss = "MSE", optimizer = "adam")
#model.fit(szenario_data, szenarios, batch_size=32, epochs = 200, verbose = True, shuffle=1)
#
#
#from data_utils import get_CIFAR10_data
#data_dict = get_CIFAR10_data()
#
#X_train = data_dict["X_train"]
#y_train = data_dict["y_train"]
#X_val
#y_val
#X_test
#y_test
## convert class vectors to binary class matrices
#num_classes = 10
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_val = keras.utils.to_categorical(y_val, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#
#print('Train data shape: ', X_train.shape)
#print('Train labels shape: ', y_train.shape)
#print('Validation data shape: ', X_val.shape)
#print('Validation labels shape: ', y_val.shape)
#print('Test data shape: ', X_test.shape)
#print('Test labels shape: ', y_test.shape)
# Invoke the above function to get our data.

# =============================================================================
# S = 1
# d_lambda = 0
# for alpha in sz_alpha:
#     for beta in sz_beta:
#         for gamma_star in sz_gamma:       
#             for omega in sz_omega:                         
#                 vola = HNG_MC_simul(alpha, beta, gamma_star, omega, d_lambda, S, K, r, Maturity, dt, output=1)
#                 #szenario_vola_calls[(alpha,beta,gamma_star,omega)] =  vola.reshape((1,vola.shape[0]*vola.shape[1]))
#                 szenario_data.append(np.concatenate((np.asarray([alpha,beta,gamma_star,omega]).reshape((1,4)),vola.reshape((1,vola.shape[0]*vola.shape[1]))),axis=1))   
# 
# # #===========================================================================
# #  
# =============================================================================
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
