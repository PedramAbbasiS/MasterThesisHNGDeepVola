 for i = 1:12000
    disp(i)
    S_0=0.5;
    X=0.3;                      
    Sig_=.04/252;              
    T=30;                    
    r=.05/365;
    price = HestonNandi(S_0,X,Sig_,T,r);
end
disp(price)
