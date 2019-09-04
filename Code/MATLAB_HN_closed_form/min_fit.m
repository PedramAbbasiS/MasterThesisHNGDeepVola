function y = min_fit(Sig_,K,S,Maturity,r,w,a,b,g,surface)
for k = 1:length(K)
    for t=1:length(Maturity)
        prices(k,t)= HestonNandi(S,K(k),Sig_,Maturity(t),r/365,w,a,b,g,0);                
    end
end
%y=sum(sum(abs(prices-surface).^2));%/abs(prices_bs)));
y=mean(mean(abs(prices-surface)./surface));%/abs(prices_bs)));

end