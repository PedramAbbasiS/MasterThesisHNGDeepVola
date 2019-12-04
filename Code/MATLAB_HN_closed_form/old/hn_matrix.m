function y = hn_matrix(Sig_,K,S,Maturity,r,w,a,b,g)
y = zeros(length(K),length(Maturity));
for k = 1:length(K)
    for t=1:length(Maturity)
        y(k,t)= HestonNandi(S,K(k),Sig_,Maturity(t),r/365,w,a,b,g,0);                
    end
end
end