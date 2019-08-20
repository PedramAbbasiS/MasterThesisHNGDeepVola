#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:03:25 2019

@author: Lukas
"""
#Source: http://vollib.org/documentation/python/0.1.5/index.html 
import py_vollib.black_scholes.implied_volatility as vol
S = 100
K = 100
sigma = .2
r = .01
flag = 'c'
t = .5
price = vol.black_scholes(flag, S, K, t, r, sigma)
iv = vol.implied_volatility(price, S, K, t, r, flag)