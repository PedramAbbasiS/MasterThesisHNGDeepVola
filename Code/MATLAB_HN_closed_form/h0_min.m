%minimizer
clearvars;close all;clc;
Maturity = 30;
S = 1;
K = 0.7:0.05:1.30;
K = K*S;
r = 0.005;
prices_opt =-ones(1,length(K));
while any(any(prices_opt < 0)) || any(any(prices_opt > 0.35))
    a = 1;b = 1;g = 1;
    while (b+a*g^2 >= 1)
        a = 5.8e-7 + (1.4e-6-5.8e-7).*rand(1,1);
        b = .43 + (.75-.43).*rand(1,1);
        g = 441 + (590-441).*rand(1,1);
        disp(b+a*g^2);
    end
    w = 4.1e-7 + (2.9e-6-4.1e-7).*rand(1,1);
    sig0 =(w+a)/(1-a*g^2-b);
    args=fmincon(@(sig) error1(sig,K,S,Maturity,r,w,a,b,g),sig0,[],[],[],[],0.000001,0.001);
    prices_opt = zeros(1,length(K));
    for k = 1:length(K)
        prices_opt(k)= HestonNandi(S,K(k),args,Maturity,r/365,w,a,b,g,0);
    end
    disp(min(prices_opt));
end

%%
prices_bs = zeros(1,length(K));
prices_hng = zeros(1,length(K));
iv = zeros(1,length(K));
for k = 1:length(K)
    prices_hng(k)= HestonNandi(S,K(k),(w+a)/(1-a*g^2-b),Maturity,r/365,w,a,b,g,0);
    prices_bs(k)= blsprice(S,K(k),r,Maturity/252,sqrt(252*(w+a)/(1-a*g^2-b)));
end
for k = 1:length(K)%min(length(K(prices3>0)),length(K(prices1>0)))
    iv(k) = blsimpv(S,K(k),r,Maturity/252,prices_opt(k),10,0,1e-15,true);
end
figure
plot(K,prices_opt);hold on,plot(K,prices_bs);hold on,plot(K,prices_hng);
set(gca,'YScale','log')
legend('optHNG','BS','HNG')

figure
plot(K,iv)
legend('optHNG')

%%
function y = error1(Sig_,K,S,Maturity,r,w,a,b,g)
        for k = 1:length(K)
            prices1(k)= HestonNandi(S,K(k),Sig_,Maturity,r/365,w,a,b,g,0);
            prices2(k)=blsprice(S,K(k),r,Maturity/252,sqrt(252*Sig_));%(w+a)/(1-a*g^2-b)));
        end
        y=sum(abs(prices1-prices2)./abs(prices1));
end
