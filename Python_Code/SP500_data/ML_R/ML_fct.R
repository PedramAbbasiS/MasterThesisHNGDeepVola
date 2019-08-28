loglik_heston<-function(para,x,r=0.00*rep(1, length(x))){
  # "para" is a vector containing the parameters over
  # which the optimization will be performed
  # "x" is a vector containing the historical returns on the underlying asset # r is the risk-free rate, expressed here on a yearly basis
  a0=para[1] # the variance’s intercept 
  b1=para[2] # the persistence parameter 
  a1=para[3] # the autoregressive parameter 
  gamma=para[4] # the leverage parameter 
  lambda0=para[5] # the risk premium
  # the log-likelihood is initialized at 0 
  loglik=0
  # The first value for the variance is set to be equal to its long term value 
  h=(a0+a1)/(1-b1-a1*gamma^2)
  # The next for loop recursively computes the conditional variance, 
  # risk premium and the associated density

  for (i in 1:length(x)){
    # The conditional log-likelihood at time i is:
    temp=dnorm(x[i],mean=r[i]/250+lambda0*h,sd=sqrt(h),log=TRUE) # The full log-likelihood is then obtained by summing up
    # those individual log-likelihood
    loglik=loglik+temp
    # The epsilon is then computed: 
    eps=x[i]-(r[i]/250+lambda0*h)
    # An the conditional variance is updated as well
    h=a0+a1*(eps/sqrt(h)-gamma*sqrt(h))^2+b1*h }
  # R provides minimizers, that is why the maximum likelihood is # obtained by trying to minimize -loglik
  return(-loglik)
}

data = read.table("SP500_data.txt", header = TRUE, sep = ",")
idx = ((data[,2]>= 1990) & (data[,2]<= 2010))
#idx = (data[,2]==1950)
logret = data[idx,4]
initial_parameters = c(5e-6,0.59,1e-6,420,0.2)
result = optim(initial_parameters, loglik_heston, x=logret, constrol = )