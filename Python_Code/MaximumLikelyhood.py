"""
MLE for HNG(1,1)

Theta = (omega,alpha,beta,gamma,lambda)
"""
import numpy as np
from scipy.optimize import minimize
from datetime import date

dates =[]
data = np.loadtxt("C:/Users/henri/Documents/SeminarOptions/Python_Code/SP500_data/SP500_data.txt",delimiter=',')
for i in range(data.shape[0]):
    data[i,0]-=366 # conversion formula matlab to python
    dates.append(date.fromordinal(int(data[i,0])))

idx = data[:,1]==2012
logret = data[idx,3]
# Loglikelyhood with normal error

#initialised with uncond. variance
def ll_hng_n(par0,x,r=0):
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
    for i in range(len(x)):
        # The conditional log-likelihood at time i is:
        temp = -0.5*np.log(2*np.pi)-0.5*np.log(h)-0.5*(x[i]-r-lambda0*h)**2/h
        #temp=dnorm(x[i],mean=r+lambda0*h,sd=sqrt(h),log=TRUE)
        loglik=loglik+temp
        # An the conditional variance is updated as well
        h=omega+alpha*(x[i]-r-(lambda0+gamma)*h)**2/h+beta*h
    return -loglik
             
         
         
         
  

#stationarity constraintro encounter
hng_stat = {'type': 'ineq','fun': lambda x: 1 - (x[3]**2)*x[1]-x[2]-1e-06}
#Theta = (omega,alpha,beta,gamma,lambda)
#par0 = np.array([5e-6,1e-6,0.6,200,0.2])
par0 = np.array([0.01,0.001,0.6,20,0.2])
#par0 are choosen as hng values from 1992 
result = minimize(ll_hng_n, par0, method='SLSQP',args = (logret), constraints =  hng_stat,
                  bounds = [(1e-08,1),(0.0,1.0),(0,1.0),(-1000,1000),(-10,10)],options={'disp': True,'ftol':1e-10})


