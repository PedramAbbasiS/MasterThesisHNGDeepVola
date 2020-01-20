""" List of different usefull functions:
    custom errors, datatransformations, etc
"""

# Sclaes Parameter to be between -1 and 1
def myscale(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=(2*x[i] - (ub[i] + lb[i])) / (ub[i] - lb[i])
    return res
def myinverse(x):
    res=np.zeros(Nparameters)
    for i in range(Nparameters):
        res[i]=x[i]*(ub[i] - lb[i]) *0.5 + (ub[i] + lb[i])*0.5
    return res

# calculates number of constraint violations for parameter combinations
def constraint_violation(x):
    return np.sum(x[:,0]*x[:,2]**2+x[:,1]>=1)/x.shape[0],x[:,0]*x[:,2]**2+x[:,1]>=1,x[:,0]*x[:,2]**2+x[:,1]  
def convex_violation(y_pred):
    error = np.zeros(y_pred.shape[0])
    for i in range(Nmaturities):
        if (i==0 or i==Nmaturities-1):
            for j in range(1,Nstrikes-1):
                convexity = y_pred[:,0,i,j]>0.5*(y_pred[:,0,i,j-1]+y_pred[:,0,i,j+1])       
                error +=   convexity

        else: 
            for j in range(1,Nstrikes-1):
                convexity = np.logical_or(y_pred[:,0,i,j]>0.5*(y_pred[:,0,i-1,j]+y_pred[:,0,i+1,j]),y_pred[:,0,i,j]>0.5*(y_pred[:,0,i,j-1]+y_pred[:,0,i,j+1]))       
                error +=   convexity

    for j in [0,Nstrikes-1]:
        for i in range(1,Nmaturities-1):
            convexity = y_pred[:,0,i,j]>0.5*(y_pred[:,0,i-1,j]+y_pred[:,0,i+1,j])     
            error +=   convexity
    mean_error = np.mean(error)
    return  error,mean_error    
       


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))   
def root_relative_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))    

def rmse_constraint(param):
    def rel_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))  +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint

def mse_constraint(param):
    def rel_mse_constraint(y_true, y_pred):
            traf_a = 0.5*(y_pred[:,0]*diff[0]+bound_sum[0])
            traf_g = 0.5*(y_pred[:,2]*diff[2]+bound_sum[2])
            traf_b = 0.5*(y_pred[:,1]*diff[1]+bound_sum[1])
            constraint = traf_a*K.square(traf_g)+traf_b
            #constraint = K.variable(value=constraint, dtype='float64')
            return tf.keras.losses.MSE( y_true, y_pred) +param*K.mean(K.greater(constraint,1))
    return rel_mse_constraint


def root_relative_mean_squared_error_convexity(param):
    def help_error(y_true, y_pred):
        error = K.sqrt(K.mean(K.square((y_pred - y_true)/y_true)))
        for i in range(1,Nmaturities-1):
            for j in range(1,Nstrikes-1):
                convexity = K.mean(K.control_flow_ops.math_ops.logical_or(K.greater(y_pred[:,0,i,j],0.5*(y_pred[:,0,i-1,j]+y_pred[:,0,i+1,j])),K.greater(y_pred[:,0,i,j],0.5*(y_pred[:,0,i,j-1]+y_pred[:,0,i,j+1]))))       
                error = error+param*convexity/((Nstrikes-1)*(Nmaturities-1))
        return error
    return help_error