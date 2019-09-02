"""
MLE for HNG(1,1)

Theta = (omega,alpha,beta,gamma,lambda)
"""
import numpy as np
from scipy.optimize import minimize
from datetime import date
from help_fun import ll_hng_n
dates =[]
data = np.loadtxt("C:/Users/henri/Documents/SeminarOptions/Python_Code/SP500_data/SP500_data.txt",delimiter=',')
for i in range(data.shape[0]):
    data[i,0]-=366 # conversion formula matlab to python
    dates.append(date.fromordinal(int(data[i,0])))


    
# Opti Params:
#params = np.zeros((4,5))
l = 0
par0 = np.array([1.8e-06,1.3e-6,0.67,448,13])

#for i in [2012,2013,2014,2015]:
idx = data[:,1]==2012#i
logret = data[idx,3]
#result = minimize(ll_hng_n, par0, method='Nelder-Mead',args = (logret),options={'disp': True,'ftol': 1e-6})
hng_stat = [{'type': 'ineq','fun': lambda x: 1 - (x[3]**2)*x[1]-x[2]-1e-8},{'type': 'ineq','fun':lambda x: x[2]},{'type': 'ineq','fun':lambda x:x[1]},{'type': 'ineq','fun':lambda x:x[0]-1e-8}]
#boundaries = [(0,100,True),(0,100,True),(0,100,True),(-10000,10000,True),(-10000,10000,True)]
#boundaries =[]
#result = minimize(ll_hng_n, par0, method='COBYLA',args = (logret), constraints = hng_stat, options={'disp': True,'tol':1e-5,'rhobeg': (1e-6,1e-6,0.1,2,0.5),'catol':(0,0,0),'maxiter':1000})
result = minimize(ll_hng_n, par0, method='SLSQP',args = (logret),constraints=hng_stat,options={'disp': True,'ftol':1e-8})
print(result.x)
#params[l,:]=result.x
if ((result.x[3]**2)*result.x[1]-result.x[2])<1:
    par0 = result.x
l += 1

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


