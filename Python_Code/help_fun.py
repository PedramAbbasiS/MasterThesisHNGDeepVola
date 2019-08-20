import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import hngoption as hng

def HNG_MC(alpha, beta, gamma, omega, lam, S_0, K, rate, T, dt, PutCall = 2, num_path = int(1e6), 
           risk_neutral = True, Variance_specs = "unconditional"):
    """
    This function calculates the Heston-Nandi-GARCH(1,1) option price of european calls/puts with MonteCarloSim
    Requirements: numpy
    Model Parameters (riskfree adjust will be done automatically): 
        alpha,beta,gamma,omega,lam
    Underlying:
        S_0 starting value, K np.array of Strikes, dt Timeshift, T maturity in dt, r riskfree rate in dt
    Function Parameters:
        Putcall: option type (1=call,0=put,2=both)
        num_path: number of sim paths
        risk_neutral: Baysian, type of simulation
        Variance_specs: Type of inital variance or inital variance input
    
    (C) Henrik Brautmeier, Lukas Wuertenberger 2019
    """
    
    gamma_star = gamma+lam+0.5
    
    # Variance Input =========================================================
    if Variance_specs=="uncondtional":
        V = (omega+alpha)/(1-alpha*gamma**2-beta)
    elif Variance_specs=="uncondtional forecast":
        sigma2=(omega+alpha)/(1-alpha*gamma_star**2-beta)
        V = omega+alpha-2*gamma_star*np.sqrt(sigma2)+(beta+gamma_star**2)*sigma2
    elif any(type(Variance_specs)==s for s in [float,int,type(np.array([0])[0]),type(np.array([0.0])[0])]):
        V = Variance_specs  #checks if Variance_specs is a float,int,numpy.int32 or numpy.float64
    else:
        print("Variance format not recognized. Uncondtional Variance will be used")
        V = (omega+alpha)/(1-alpha*gamma**2-beta) 
    # ========================================================================
    
    # Initialisation
    r = np.exp(rate*dt)-1                             
    lnS = np.zeros((num_path,T+1))
    h = np.zeros((num_path,T+1))
    lnS[:,0] = np.log(S_0)*np.ones((num_path))
    h[:,0] = V*np.ones((num_path)) #initial wert
    z = np.random.normal(size=(num_path,T+1))
    
    # Simulation
    for t in np.arange(dt,T+dt,dt):
        if risk_neutral:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma_star*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r-0.5*h[:,t]+np.sqrt(h[:,t])*z[:,t]
        else:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r+lam*h[:,t]+np.sqrt(h[:,t])*z[:,t]
    S_T = np.exp(lnS[:,-1])
    
    # Output
    if PutCall==1: # Call
        return np.exp(-rate*T)*np.mean(np.maximum(S_T[:,np.newaxis] - K,np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
    elif PutCall==0: # Put
        return np.exp(-rate*T)*np.mean(np.maximum(K-S_T[:,np.newaxis],np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
    elif PutCall==2: # (Call,Put)
        return np.exp(-rate*T)*np.mean(np.maximum(S_T[:,np.newaxis] - K,np.zeros((S_T.shape[0],K.shape[0]))),axis=0),np.exp(-rate*T)*np.mean(np.maximum(K-S_T[:,np.newaxis],np.zeros((S_T.shape[0],K.shape[0]))),axis=0)