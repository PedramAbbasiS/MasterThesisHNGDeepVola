#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 08:52:37 2019

@author: Lukas
"""

import hngoption as hng


PutCall = 1
d_lambda = 2                    #real
d_lambda_ = -.5                 #risk-free
alpha = 0.000005
beta = 0.85
gamma = 150                     #real
gamma_ = gamma+d_lambda+.5      #risk-free
T = 30
S = .5
K = .3
r = 0.05/365
sigma = 0.04/252
omega = sigma * (1-beta-alpha*gamma**2)-alpha 
price = hng.HNC(alpha, beta, gamma_, omega, d_lambda_, sigma, S, K, r, T, PutCall)
print(price)