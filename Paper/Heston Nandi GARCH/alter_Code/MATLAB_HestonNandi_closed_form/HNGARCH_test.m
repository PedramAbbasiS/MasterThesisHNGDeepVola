S_0=1;                    
X=0.9;
w = 0.0001;
g = 0.002;
g_=g+lam+.5;
b = 0.001;
a = 0.001;
lam = -0.0005;
sigma2 = w/(1-a*g_^2-b);
Sig_= w + b*sigma2+a*(-lam*sigma2-g_*sigma2)^2/sigma2;           
T=10;                       
r=0;                  
HestonNandi(S_0,X,Sig_,T,r)
%f = HestonNandi1(S_0,X,Sig_,T,r)