"""
MLE for HNG(1,1)

Theta = (omega,alpha,beta,gamma,lambda)
"""
import numpy as np
# Loglikelyhood with normal error
#initialised with uncond. variance

def ll_hng_n(par0,x,r, out=None):
    T = np.size(x,axis = 0)
    omega = par0[0]
    alpha =par0[1]
    beta = par0[2]
    gamma = par0[3]
    lam0 = par0[4]
    lag = 1
    h =np.ones((T+lag,1))*(alpha+omega)/(1-alpha*gamma**2-beta)
    x = np.append(np.zeros((lag,1)),x).reshape(T+lag,1)  
    for t in range(lag,T+lag):
            h[t]=omega+alpha+(alpha*gamma**2+beta)*h[t-1]
    h = h[lag:]
    #take minus loglikely for maximization
    ll_vec = np.log(2*np.pi)/2 + (np.log(h))/2 + (np.divide((x[lag:]-r-lam0*h)**2,2*h))
    ll = np.sum(ll_vec)
    if out is None:
        return ll
    else:
        return ll,h,ll_vec

def loglik_heston_n(par0,x,r):
    # "para" is a vector containing the parameters over# which the optimization will be performed
    # "x" is a vector containing the historical returns on the underlying asset
    # r is the risk-free rate, expressed here on a daily basis
    a0=par0[0] # the variance’s intercept
    b1=par0[2] # the persistence parameter
    a1=par0[1] # the autoregressive parameter
    gamma=par0[3] # the leverage parameter
    lambda0=par0[4] # the risk premium
    # the log-likelihood is initialized at 0
    loglik=0
    # The first value for the variance is set to be equal to its long term value
    h=(a0+a1)/(1-b1-a1*gamma**2)
    # The next for loop recursively computes the conditional variance, risk premium 
    #and the associated density
    for i in range(1,len(x)):
        # The conditional log-likelihood at time i is:
        temp = -0.5*(np.log(h)-(x[i]-r-lambda0*h)**2/h)
        #temp=dnorm(x[i],mean=r+lambda0*h,sd=sqrt(h),log=TRUE)
        # The full log-likelihood is then obtained by summing up
        # those individual log-likelihood
        loglik=loglik+temp
        # The epsilon is then computed:
        eps=x[i]-(r+lambda0*h)
        # An the conditional variance is updated as well
        h=a0+a1*(eps/np.sqrt(h)-gamma*np.sqrt(h))**2+b1*h
        # R provides minimizers, that  is why the maximum likelihood is
        # obtained by trying to minimize -loglik
    return -loglik
             
         
         
         
         
         
# Pseudo loglikelyhood with t-distr error
#initialised with uncond. variance
#def ll_hng_t(par0,x, out=None):
#    T = np.size(x,axis = 0)
#    c = par0[0]
#    arch = par0[1]
#    garch = par0[2]
#    v = par0[3]
#    lag = 1
#    s2 =np.ones((T+lag,1))*c/(1-arch-garch)
#    x = np.append(np.zeros((lag,1)),x).reshape(T+lag,1)  
#    for t in range(lag,T+lag):
#        s2[t]=c+ arch*x[t-1]**2+ garch*s2[t-1]
#    s2 = s2[lag:]
#    g1 = gamma((v+1)/2)
#    g2 = gamma(v/2)
#    lls = -np.log(g1/(np.sqrt(np.pi)*g2*np.sqrt(v-2))) + (np.log(s2))/2+((v+1)/2)*(np.log(1+np.divide(x[lag:]**2,s2)/(v-2)))
#    ll = np.sum(lls)
#    if out is None:
#        return ll
#    else:
#        return ll,s2,lls

#stationarity constraint
hng_stat = {'type': 'ineq','fun': lambda x: 1 - (x[3]**2)*x[1]-x[2]-1e-08}
#Theta = (omega,alpha,beta,gamma,lambda)
par0 = np.array([5e-6,1e-6,0.6,400,0.2])
#par0 are choosen as hng values from 1992 
result = minimize(ll_hng_n, par0, method='SLSQP',args = (insample*100), constraints =  hng_stat,
                  bounds = [(1e-08,1),(0.0,1.0),(-1.0,1.0),(-1000,1000),()],options={'disp': True,'ftol':1e-10})


