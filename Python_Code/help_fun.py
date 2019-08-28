import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import hngoption as hng
#import py_vollib.black_scholes.implied_volatility as vol!
from calcbsimpvol import calcbsimpvol 

def data_generator(sz_alpha,sz_beta,sz_gamma,sz_omega,K,Maturity,dt=1,r=0,value=1,form=1,typ="MC"):
    szenario_data =[]
    if not(form):
        szenarios = []
    for alpha in sz_alpha:
        for beta in sz_beta:
            for gamma_star in sz_gamma:       
                for omega in sz_omega:     
                    if typ=="MC":                    
                        data = HNG_MC_simul(alpha, beta, gamma_star, omega, 0, 1, K, r, Maturity, dt, output=value)
                    elif typ=="Model":
                        sigma2 = (omega+alpha)/(1-alpha*gamma_star**2-beta) 
                        data = np.zeros((Maturity.shape[0],K.shape[0]))
                        l = 0
                        for k in K:
                            s = 0
                            for T in Maturity:
                                data[s,l]= hng.HNC(alpha, beta, gamma_star, omega, -0.5, sigma2, 1, k, 0, T, 1)
                                s+=1
                            l+=1
                    if form:
                        szenario_data.append(np.concatenate((np.asarray([alpha,beta,gamma_star,omega]).reshape((1,4)),data.flatten()),axis=1))   
                    else:
                        szenario_data.append(data)
                        szenarios.append([alpha,beta,gamma_star,omega])
    
    szenario_data = np.asarray(szenario_data)
    if form:
        szenario_data = szenario_data.reshape(szenario_data.shape[0],szenario_data.shape[-1])
        return szenario_data
    else:
        szenarios = np.asarray(szenarios)
        return szenario_data,szenarios
    

def HNG_MC(alpha, beta, gamma, omega, d_lambda, S, K, rate, T, dt, PutCall = 1, num_path = int(1e5), 
           risk_neutral = True, Variance_specs = "unconditional",output="1"):
    """
    This function calculates the Heston-Nandi-GARCH(1,1) option price of european calls/puts with MonteCarloSim
    Requirements: numpy
    Model Parameters (riskfree adjust will be done automatically): 
        under P: alpha,beta,gamma,omega,d_lambda
        under Q: alpha,beta,gamma_star,omega 
    Underlying:
        S starting value, K np.array of Strikes, dt Timeshift, T maturity in dt, r riskfree rate in dt
    Function Parameters:
        Putcall: option type (1=call,-1=put,2=both)
        num_path: number of sim paths
        risk_neutral: Baysian, type of simulation
        Variance_specs: Type of inital variance or inital variance input
        output: output spec, 0=bs imp vola,1=option price, 2=both
    
    (C) Henrik Brautmeier, Lukas Wuertenberger 2019, University of Konstanz
    """
  
    # Variance Input =========================================================
    if Variance_specs=="unconditional":
        V = (omega+alpha)/(1-alpha*gamma**2-beta)
    elif Variance_specs=="uncondtional forecast":
        sigma2=(omega+alpha)/(1-alpha*gamma**2-beta)
        V = omega+alpha-2*gamma*np.sqrt(sigma2)+(beta+gamma**2)*sigma2
    elif any(type(Variance_specs)==s for s in [float,int,type(np.array([0])[0]),type(np.array([0.0])[0])]):
        V = Variance_specs  #checks if Variance_specs is a float,int,numpy.int32 or numpy.float64
    else:
        print("Variance format not recognized. Uncondtional Variance will be used")
        V = (omega+alpha)/(1-alpha*gamma**2-beta) 
    
    
    # Simulation =============================================================
    #Initialisation
    r = np.exp(rate*dt)-1                             
    lnS = np.zeros((num_path,T+1))
    h = np.zeros((num_path,T+1))
    lnS[:,0] = np.log(S)*np.ones((num_path))
    h[:,0] = V*np.ones((num_path)) #initial wert
    z = np.random.normal(size=(num_path,T+1))
    
    # Monte Carlo
    for t in np.arange(dt,T+dt,dt):
        if risk_neutral:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r-0.5*h[:,t]+np.sqrt(h[:,t])*z[:,t]
        else:
            h[:,t] = omega+beta*h[:,t-dt]+alpha*(z[:,t-dt]-gamma*np.sqrt(h[:,t-dt]))**2
            lnS[:,t] = lnS[:,t-dt]+r+d_lambda*h[:,t]+np.sqrt(h[:,t])*z[:,t]
    S_T = np.exp(lnS[:,-1])
    
    
    # Output =================================================================
    # Prices
    if PutCall==1: # Call
        price = np.exp(-rate*T)*np.mean(np.maximum(S_T[:,np.newaxis] - K,np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
    elif PutCall==-1: # Put
        price = np.exp(-rate*T)*np.mean(np.maximum(K-S_T[:,np.newaxis],np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
    elif PutCall==2: # (Call,Put)
        price_call,price_put =  np.exp(-rate*T)*np.mean(np.maximum(S_T[:,np.newaxis] - K,np.zeros((S_T.shape[0],K.shape[0]))),axis=0),np.exp(-rate*T)*np.mean(np.maximum(K-S_T[:,np.newaxis],np.zeros((S_T.shape[0],K.shape[0]))),axis=0)
        price = (price_call,price_put)
    
    # Implied Vola
    if output==1:
        return price
    elif (output==0 or output==2):
        K_tmp,tau = np.meshgrid(K.reshape((K.shape[0],1)),np.array(T/252))
        if PutCall==1 or PutCall==-1:
            vola = calcbsimpvol(dict(cp=np.asarray(PutCall), P=np.asarray(price.reshape((1,price.shape[0]))), S=np.asarray(S), K=K_tmp, tau=tau, r=np.asarray(rate*252), q=np.asarray(0)))
            if output==0:
                return vola
            else:
                return price,vola
        elif PutCall==2:
            vola_call = calcbsimpvol(dict(cp=np.asarray(1), P=np.asarray(price_call.reshape((1,price_call.shape[0]))), S=np.asarray(S), K=K_tmp, tau=tau, r=np.asarray(rate*252), q=np.asarray(0)))
            vola_put = calcbsimpvol(dict(cp=np.asarray(-1), P=np.asarray(price_put.reshape((1,price_put.shape[0]))), S=np.asarray(S), K=K_tmp, tau=tau, r=np.asarray(rate*252), q=np.asarray(0)))
            vola = (vola_call,vola_put)
            if output==0:
                return vola
            else:
                return price,vola




def HNG_MC_simul(alpha, beta, gamma, omega, d_lambda, S, K, rate, T, dt, PutCall = 1, num_path = int(1e4), 
           risk_neutral = True, Variance_specs = "unconditional",output=1):
    """
    This function calculates the Heston-Nandi-GARCH(1,1) option price of european calls/puts with MonteCarloSim
    Requirements: numpy
    Model Parameters (riskfree adjust will be done automatically): 
        under P: alpha,beta,gamma,omega,d_lambda
        under Q: alpha,beta,gamma_star,omega 
    Underlying:
        S starting value, K np.array of Strikes, dt Timeshift, T np.array of maturities in dt, r riskfree rate for dt
    Function Parameters:
        Putcall: option type (1=call,-1=put,2=both)
        num_path: number of sim paths
        risk_neutral: Baysian, type of simulation
        Variance_specs: Type of inital variance or inital variance input
        output: output spec, 0=bs imp vola,1=option price, 2=both
        If implied vola is calculated you need to use dt=1
    
    (C) Henrik Brautmeier, Lukas Wuertenberger 2019, University of Konstanz
    """
  
    # Variance Input =========================================================
    if Variance_specs=="unconditional":
        V = (omega+alpha)/(1-alpha*gamma**2-beta)
    elif Variance_specs=="uncondtional forecast":
        sigma2=(omega+alpha)/(1-alpha*gamma**2-beta)
        V = omega+alpha-2*gamma*np.sqrt(sigma2)+(beta+gamma**2)*sigma2
    elif any(type(Variance_specs)==s for s in [float,int,type(np.array([0])[0]),type(np.array([0.0])[0])]):
        V = Variance_specs  #checks if Variance_specs is a float,int,numpy.int32 or numpy.float64
    else:
        print("Variance format not recognized. Uncondtional Variance will be used")
        V = (omega+alpha)/(1-alpha*gamma**2-beta) 
    
    
    # Simulation =============================================================
    #Initialisation
    T_max = np.max(T)
    r = np.exp(rate*dt)-1                             
    lnS = np.zeros((num_path,T_max+1))
    h = np.zeros((num_path,T_max+1))
    lnS[:,0] = np.log(S)*np.ones((num_path))
    h[:,0] = V*np.ones((num_path)) #initial wert
    z = np.random.normal(size=(num_path,T_max+1))

    # Monte Carlo
    for t in np.arange(1,T_max+1,1):
        if risk_neutral:
            h[:,t] = omega+beta*h[:,t-1]+alpha*(z[:,t-1]-gamma*np.sqrt(h[:,t-1]))**2
            lnS[:,t] = lnS[:,t-1]+r-0.5*h[:,t]+np.sqrt(h[:,t])*z[:,t]
        else:
            h[:,t] = omega+beta*h[:,t-1]+alpha*(z[:,t-1]-gamma*np.sqrt(h[:,t-1]))**2
            lnS[:,t] = lnS[:,t-1]+r+d_lambda*h[:,t]+np.sqrt(h[:,t])*z[:,t]
    matS = np.exp(lnS[:,T])
    
    # Output =================================================================
    # Prices
    m = T.shape[0]
    n = K.shape[0]
    if PutCall!=2:
        price = np.zeros((m,n))
    else:
        price_call = np.zeros((m,n))
        price_put = np.zeros((m,n))
      
    for t in range(m):
        S_t = matS[:,t]
        if PutCall==1: # Call
            price[t,:] = np.exp(-rate*T[t])*np.mean(np.maximum(S_t[:,np.newaxis] - K,np.zeros((S_t.shape[0],n))),axis=0)
        elif PutCall==-1: # Put
            price = np.exp(-rate*T[t])*np.mean(np.maximum(K-S_t[:,np.newaxis],np.zeros((S_t.shape[0],n))),axis=0)
        elif PutCall==2: # (Call,Put)
            price_call,price_put =  np.exp(-rate*T[t])*np.mean(np.maximum(S_t[:,np.newaxis] - K,np.zeros((S_t.shape[0],n))),axis=0),np.exp(-rate*T[t])*np.mean(np.maximum(K-S_t[:,np.newaxis],np.zeros((S_t.shape[0],n))),axis=0)
            price = (price_call,price_put)
        
    # Implied Vola
    if output==1:
        return price
    elif (output==0 or output==2):
        T = T*dt #normalisation to yearly basis
        K_tmp,tau = np.meshgrid(K.reshape((n,1)),T.reshape((m,1)))
        if PutCall==1 or PutCall==-1:
            vola = calcbsimpvol(dict(cp=np.asarray(PutCall), P=price, S=np.asarray(S), K=K_tmp, tau=tau, r=np.asarray(rate/dt), q=np.asarray(0)))
            if output==0:
                return vola
            else:
                return price,vola
        elif PutCall==2:
            vola_call = calcbsimpvol(dict(cp=np.asarray(1), P=price_call, S=np.asarray(S), K=K_tmp, tau=tau, r=np.asarray(rate/dt), q=np.asarray(0)))
            vola_put = calcbsimpvol(dict(cp=np.asarray(-1), P=price_put, S=np.asarray(S), K=K_tmp, tau=tau, r=np.asarray(rate/dt), q=np.asarray(0)))
            vola = (vola_call,vola_put)
            if output==0:
                return vola
            else:
                return price,vola

def ll_hng_n(par0,x,r=0,history=False):
    # "par0" is a vector containing the parameters over# which the optimization will be performed
    # "x" is a vector containing the historical log returns on the underlying asset
    # "r" is the risk-free rate, expressed here on a daily basis
    omega=par0[0]   # the variance’s intercept
    alpha=par0[1]   # the autoregressive parameter
    beta=par0[2]    # the persistence parameter
    gamma=par0[3]   # the leverage parameter
    lambda0=par0[4] # the risk premium
    loglik=0        # the log-likelihood is initialized at 0
    h=(omega+alpha)/(1-beta-alpha*gamma**2) # initianlize variance as longterm variance
    h2 = np.zeros((len(x)+1,1))
    h2[0] = h
    for i in range(0,len(x)):
        # The conditional log-likelihood at time i is:
        #temp = -0.5*np.log(2*np.pi)-0.5*np.log(h)-0.5*(x[i]-r-lambda0*h)**2/h
        temp = stats.norm.logpdf(x[i],r+lambda0*h,np.sqrt(h))
        loglik=loglik+temp
        # An the conditional variance is updated as well
        h=omega+alpha*(x[i]-r-(lambda0+gamma)*h)**2/h+beta*h
        h2[i+1] = h
    if history:
        return -loglik,h2
    else:
        return -loglik










"""


todo + fragen
    bs input were sind jährlich? unser input is täglich? lösung1: daily convertion *252 (implementiert)
    lösung2=yearly vola output convertieren? (nicht implementiert,geht das überhaupt?)
     



# BS implied vola TEStING LABRATORY        
# beispiel des erstellers    
#für documentation calcbsimpvol öffnen
S = np.asarray(100)         #Spotprice
K_value = np.arange(40, 160, 25)
K = np.ones((np.size(K_value), 1))
K[:, 0] = K_value           #strike
tau_value = np.arange(0.25, 1.01, 0.25)
tau = np.ones((np.size(tau_value), 1))
tau[:, 0] = tau_value       #T-t
r = np.asarray(0.01)        #riskfree
q = np.asarray(0.03)        #dividen yield
cp = np.asarray(1)          #call=1 put =-1
P = [[59.35, 34.41, 10.34, 0.50, 0.01],
[58.71, 33.85, 10.99, 1.36, 0.14],
[58.07, 33.35, 11.50, 2.12, 0.40],
[57.44, 32.91, 11.90, 2.77, 0.70]]
#optionalprice matrix for maturity x K
P = np.asarray(P)
[K, tau] = np.meshgrid(K, tau)

sigma = calcbsimpvol(dict(cp=cp, P=P, S=S, K=K, tau=tau, r=r, q=q))
print(sigma)    
 """   