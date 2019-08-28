"""
MLE for HNG(1,1)

Theta = (omega,alpha,beta,gamma,lambda)
"""
import numpy as np
from scipy.optimize import minimize
from datetime import date
from scipy import stats
import pyopt
dates =[]
data = np.loadtxt("C:/Users/henri/Documents/SeminarOptions/Python_Code/SP500_data/SP500_data.txt",delimiter=',')
for i in range(data.shape[0]):
    data[i,0]-=366 # conversion formula matlab to python
    dates.append(date.fromordinal(int(data[i,0])))

idx = data[:,1]==2012
logret = data[idx,3]
# Loglikelyhood with normal error

#initialised with uncond. variance
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
             
         
# Opti Params:

par = np.array([6e-07,1.49e-6,0.66,460,0.64])         
         

#pyOpti  

# SciPy Opti
#stationarity constraint
#Theta = (omega,alpha,beta,gamma,lambda)
#par0 = np.array([0.01,0.0001,0.1,10,0.2])
"""
hng_stat = {'type': 'ineq','fun': lambda x: 1 - (x[3]**2)*x[1]-x[2]-1e-10}
Fully Constrained
par0 = np.array([1.8e-06,1.3e-6,0.67,448,13])
result = minimize(ll_hng_n, par0, method='SLSQP',args = (logret), constraints = hng_stat ,bounds = [(0,100,False),(0,100,False),(0,100,True),(-10000,10000,False),(-10000,10000,False)],options={'disp': True,'ftol':1e-10})
"""

"positive optimal:  par0 = np.array([1.8e-06,1.3e-6,0.67,448,13])"
#R

"non positive optimal par0=np.array([ 4.8e-06,  5.3-07, -1.23,  2021,7.44])"
#print(ll_hng_n(par0,logret))
result_unbound = minimize(ll_hng_n, par0, method='Nelder-Mead',args = (logret))


