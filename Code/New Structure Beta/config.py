# config file

### Preambel
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### Data Import
mat         = scipy.io.loadmat('data_vola_maxbounds_50000_0005_09_11_30_210.mat')
data        = mat['data_vola']
Nparameters = 5
maturities  = np.array([30, 60, 90, 120, 150, 180, 210])
strikes     = np.array([0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1])
Nstrikes    = len(strikes)   
Nmaturities = len(maturities)   
xx          = data[:,:Nparameters]
yy          = data[:,Nparameters+2:]
ub=np.amax(xx, axis=0)
lb=np.amin(xx, axis=0)
diff = ub-lb
bound_sum =ub+lb

### Trainset generation
X_train, X_test, y_train, y_test = train_test_split(
    xx, yy, test_size=0.15)#, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
   X_train, y_train, test_size=0.15)#, random_state=42)

Ntest= X_test.shape[0]
Ntrain= X_train.shape[0]
Nval= X_val.shape[0]