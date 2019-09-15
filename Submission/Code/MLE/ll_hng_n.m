function mn_loglik = ll_hng_n(par0,x,r)
n = length(x);
omega = par0(1);
alpha = par0(2);
beta = par0(3);
gamma = par0(4);
lambda = par0(5);
h = (omega+alpha)/(1-beta-alpha*gamma^2);
loglik = 0;
for i = 1:n
    loglik = loglik-1/2*log(2*pi)-0.5*log(h)-0.5*(x(i)-r-lambda*h)^2/h;
    h = omega+alpha*(x(i)-r-(gamma+lambda)*h)^2/h+beta*h;
end
mn_loglik = -loglik;