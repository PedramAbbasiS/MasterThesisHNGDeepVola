import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import hngoption as hng

def HNG_MC(alpha,beta,gamma,omega,lam,S_0,K,rate,T,dt,PutCall,num_path = int(1e6), 
           risk_neutral = True):
    sigma2 =  omega/(1-alpha*gamma**2-beta)
    r = np.exp(rate*dt)-1                             
    lnS = np.zeros((num_path,T+1))
    h = np.zeros((num_path,T+1))
    lnS[:,0] = np.log(S_0)*np.ones((num_path))
    h[:,0] = sigma2*np.ones((num_path)) #initial wert
    z = np.random.normal(size=(num_path,T+1))
    gamma_star = gamma+lam+0.5
    for t in np.arange(dt,T+dt,dt):
        if risk_neutral:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma_star*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r-0.5*h[:,t]+np.sqrt(h[:,t])*z[:,t]
        else:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r+lam*h[:,t]+np.sqrt(h[:,t])*z[:,t]
    S_T = np.exp(lnS[:,-1])
    if PutCall:
        return np.exp(-rate*T)*np.mean(np.maximum(S_T-K,0))
    else:
        return np.exp(-rate*T)*np.mean(np.maximum(K-S_T,0))  