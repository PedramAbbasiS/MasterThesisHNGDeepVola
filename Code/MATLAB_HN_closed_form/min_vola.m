function y = min_vola(Sig_,K,S,Maturity,r,w,a,b,g,surface)
vola = zeros(length(K),length(Maturity));
err = 0;
for k = 1:length(K)
    for t=1:length(Maturity)
        price= HestonNandi(S,K(k),Sig_,Maturity(t),r/365,w,a,b,g,0);
        if price<=0
            err = 1;
            continue
        end
        vola(k,t) = blsimpv(S,K(k),r,Maturity(t)/252,price,10,0,1e-8,true);
    end
end
if err ==1
    y = 1e6;
else
    y=nanmean(nanmean(abs(vola-surface).^2));%/abs(prices_bs)));
%y=mean(mean(abs(prices-surface)./surface));%/abs(prices_bs)));

end