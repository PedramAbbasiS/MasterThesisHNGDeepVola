#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:26:21 2019

@author: Lukas
"""

"""
This function calculates the price of Call option based on the GARCH 
option pricing formula of Heston and Nandi(2000). The input to the
function are: current price of the underlying asset, strike price,
unconditional variance of the underlying asset, time to maturity in days,
and daily risk free interest rate.
Source: https://www.mathworks.com/matlabcentral/fileexchange/27644-heston-nandi-option-price
Author: Ali Boloorforoosh
email:  a_bol@jmsb.concordia.ca
Date:   Nov. 1,08
"""

import numpy as np
import scipy.integrate as integrate
import pickle 

# sample inputs 
S_0=100;                    #stock price at time t
X=100;                      #strike prices
Sig_=.04/252;               #unconditional variances per day
T=30;                       #option maturity
r=.05/365;                  #daily risk free rate

def HestonNandi(S_0,X,Sig_,T,r):
    #function that returns the value for the characteristic function
    def charac_fun(phi):
        phi=np.transpose(phi)    #the input has to be a row vector
        #print(phi.size)
        #print(phi)
        #GARCH parameters
        lam=2
        lam_=-.5                   #risk neutral version of lambda
        a=.000005
        b=.85
        g=150                      #gamma coefficient
        g_=g+lam+.5                #risk neutral version of gamma
        w=Sig_*(1-b-a*g**2)-a       #GARCH intercept
        #print(phi.shape)
        A = np.zeros(T)
        B = np.zeros(T)
        #recursion for calculating A(t,T,Phi)=A_ and B(t,T,Phi)=B_
        A[T-1]=phi*r
        #print(len(phi*r))
        B[T-1]=lam_*phi+.5*phi**2
        #print(A.size)
        for i in range(2,T+1):
            A[T-i]=A[T-i+1]+phi*r+B[T-i+1]*w-.5*np.log(1-2*a*B[T-i+1])
            B[T-i]=phi*(lam_+g_)-.5*g_**2+b*B[T-i+1]+.5*(phi-g_)**2/(1-2*a*B[T-i+1])
 
        #file_pi = open('test.obj', 'w') 
        #pickle.dump(np.array2string(A), file_pi)
   
        #A[T-1]=A[0]+phi*r+B[0]*w-.5*np.log(1-2*a*B[0])                    
        #B[T-1]=phi*(lam_+g_)-.5*g_**2+b*B[0]+.5*(phi-g_)**2/(1-2*a*B[0])

        f=S_0**phi*np.exp(A[0]+B[0]*Sig_)
        return np.transpose(f) #the output is a row vector
    #function Integrand1 and Integrand2 return the values inside the 
    #first and the second integrals
    def Integrand1(phi):
        return ((X ** (-complex(0,phi))*charac_fun(complex(0,phi)+1))/(complex(0,phi))).real
    def Integrand2(phi):
        return ((X ** (-complex(0,phi))*charac_fun(complex(0,phi)))/(complex(0,phi))).real
    return .5*S_0+(np.exp(-r*T)/np.pi)*integrate.quad(lambda phi: 
        Integrand1(phi),0,100)[0]-X*np.exp(-r*T)*(.5+(1/np.pi)*
                  integrate.quad(lambda phi: Integrand2(phi),0,100)[0])        
print(HestonNandi(S_0,X,Sig_,T,r))