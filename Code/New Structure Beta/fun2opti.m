function error = fun2opti(params,val_pred,r,data_vec)
tmp = val_pred-blsimpv_vec(data_vec,r,price_Q_clear([params(4),params(1),params(2),params(3)],data_vec,r/252,params(5)));
error = sum(tmp.^2);