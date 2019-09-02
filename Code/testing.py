#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:20:57 2019

@author: Lukas
"""
berg_Nparameters = 4
berg_data =np.load('rBergomiTrainSet.txt')
berg_strikes=np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
berg_maturities=np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ])
berg_xx=berg_data[:,:berg_Nparameters]
berg_yy=berg_data[:,berg_Nparameters:]
berg_max_vola = np.max(berg_yy)
berg_min_vola = np.min(berg_yy)
berg_mean_vola = np.mean(berg_yy)
berg_median_vola = np.median(berg_yy)
berg_iv_re = berg_yy.reshape((1,40000*88))
from matplotlib import pyplot as plt 
from scipy.stats import gaussian_kde
density = gaussian_kde(berg_iv_re)
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.show()
berg_low_vola = berg_yy[(berg_yy<.07).any(axis=1)]



HNG_Nparameters = 6
HNG_data =np.load('data_test_small.npy')
HNG_strikes=np.array([0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1])
HNG_maturities=np.array([20, 50, 80, 110, 140])
HNG_xx=HNG_data[:,:HNG_Nparameters]
HNG_yy=HNG_data[:,HNG_Nparameters:]
HNG_max_vola = np.max(HNG_yy)
HNG_min_vola = np.min(HNG_yy)
HNG_mean_vola = np.mean(HNG_yy)
HNG_median_vola = np.median(HNG_yy)
HNG_iv_re = HNG_yy.reshape((1,2000*35))
HNG_low_vola = HNG_yy[(HNG_yy<.07).any(axis=1)]
density = gaussian_kde(HNG_iv_re)
xs = np.linspace(0,1,200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.show()


import py_vollib.black_scholes.implied_volatility as vol
flag = 'c'
S = 1
r = 0
price = .32
vola = vol.implied_volatility(price, S, 0.9, 140/252, r, flag)
print(vola)