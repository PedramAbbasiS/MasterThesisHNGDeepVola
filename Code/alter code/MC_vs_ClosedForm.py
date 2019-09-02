# -*- coding: utf-8 -*-
"""
@author: Henrik Brautmeier, Lukas Wuertenberger

Option Pricing in Maschine Learning
"""
# preambel
import numpy as np
import matplotlib.pyplot as plt
import hngoption as hng
from help_fun import HNG_MC_simul

#szenarios tested
# 1
#sz_alpha = [0.01,0.02]
#sz_gamma = [0.2,0.3]
#sz_beta = [0.2,0.5]
#sz_omega =[0.1,0.2]
#Maturity = np.array([30,40,50,60,100,252])

#2
#sz_alpha = [0.01,0.02]
#sz_gamma = [3,5]
#sz_beta = [0.2,0.4]
#sz_omega =[0.01,0.02]
#Maturity = np.array([20,40,60,80,100,120]) 

#3
#sz_alpha = [0.01,0.02]
#sz_gamma = [0.2,0.3]
#sz_beta = [0.2,0.5]
#sz_omega =[0.1,0.2]
#Maturity = np.array([20,40,60,80,100,120]) 

#4
#sz_alpha = [0.01,0.02]
#sz_gamma = [3,5]
#sz_beta = [0.2,0.4]
#sz_omega =[0.1,0.2]
#Maturity = np.array([20,40,60,80,100,120]) 

#5 
#sz_alpha = [0.01,0.02]
#sz_gamma = [3,5]
#sz_beta = [0.2,0.4]
#sz_omega =[0.1,0.2]
#Maturity = np.array([40,80,120,160,200,240]) 

# Szenario Analyse
sz_alpha = [0.01,0.02]
sz_gamma = [3,5]
sz_beta = [0.2,0.4]
sz_omega =[0.1,0.2]
Maturity = np.array([40,80,120,160,200,240]) 

import time
szenario_data =[]
szenarios = []
szenario_model = []
dt = 1/252 # stepwidth in basis of 1year
K = np.array([0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1])
r = 0 # yearl rate times dt example 5%*1/252
for alpha in sz_alpha:
    for beta in sz_beta:
        for gamma_star in sz_gamma:       
            for omega in sz_omega:     
                
                
                #start_time = time.time()
                #data = HNG_MC_simul(alpha, beta, gamma_star, omega, 0, 1, K, r, Maturity, dt, output=1,num_path=int(1e5))
                #print("--- %s seconds ---" % (time.time() - start_time))
                #szenario_data.append(data)
                #szenarios.append([alpha,beta,gamma_star,omega])
                sigma2 = (omega+alpha)/(1-alpha*gamma_star**2-beta) 
                price= np.zeros((6,9))
                l = 0
                start_time = time.time()
                for k in K:
                    s = 0
                    for T in Maturity:
                        price[s,l]= hng.HNC(alpha, beta, gamma_star, omega, -0.5, sigma2, 1, k, 0, T, 1)
                        s+=1
                    l+=1
                print("--- %s seconds ---" % (time.time() - start_time))
                
                szenario_model.append(price)
                
from calcbsimpvol import calcbsimpvol                 
K_tmp,tau = np.meshgrid(K.reshape((K.shape[0],1)),np.array(Maturity/252))
price = szenario_model1[15,:,:]
vola = calcbsimpvol(dict(cp=np.asarray(1), P=price, S=np.asarray(1), K=K_tmp, tau=tau, r=np.asarray(0), q=np.asarray(0)))
price = szenario_data1[15,:,:]
vola2 = calcbsimpvol(dict(cp=np.asarray(1), P=price, S=np.asarray(1), K=K_tmp, tau=tau, r=np.asarray(0), q=np.asarray(0)))
                         
                
#Saving szenario
#szenario_model5 = np.asarray(szenario_model)
#szenario_data5 = np.asarray(szenario_data)
#szenarios5 = np.asarray(szenarios)
#rel_error_matrix5 =np.divide(np.abs(szenario_data5-szenario_model5),szenario_model5)
#plt.boxplot(rel_error_matrix5.reshape((864,1)))
#print(np.median(rel_error_matrix5))

#szenario_errors = np.concatenate((szenario_model1,szenario_model2,szenario_model3,szenario_model4,szenario_model5),axis=0)
#szenario_rel_error = np.concatenate((rel_error_matrix1,rel_error_matrix2,rel_error_matrix3,rel_error_matrix4,rel_error_matrix5),axis=0)
#szenario_mat = np.concatenate((szenarios1,szenarios2,szenarios3,szenarios4,szenarios5),axis=0)
#Maturity_vec =  np.concatenate((np.array([30,40,50,60,100,252]),np.array([20,40,60,80,100,120]),np.array([20,40,60,80,100,120]),np.array([20,40,60,80,100,120]),np.array([40,80,120,160,200,240])))
# Analyse:

# Maturity analyse:

#bucket_low =np.asarray(np.where(Maturity_vec<100))
#bucket_high=np.asarray(np.where(Maturity_vec>=100)) 
#
#
#szenario_rel_error[bucket_high,:,:].flatten()














# Interation Testing =========================================================

#model_val = hng.HNC(0.1, 0.2, 2, 0.1, -0.5, 0.2/(1-0.4-0.2), 1, 0.9, 0, 30, 1)
#test=np.zeros((10,1))
#test2=np.zeros((10,1))
#
#for i in range(10):
#    test[i]=HNG_MC_simul(0.1, 0.2, 2, 0.1, 0, 1, np.asarray([0.9]), 0, np.asarray([30]), dt, output=1,num_path=int(1e5))
#    test2[i]=HNG_MC_simul(0.1, 0.2, 2, 0.1, 0, 1, np.asarray([0.9]), 0, np.asarray([30]), dt, output=1,num_path=int(1e4))
#
#mc1=test.mean()
#mc2=test2.mean()
#print(mc1,mc2,model_val)
#print(test.std(),test2.std())