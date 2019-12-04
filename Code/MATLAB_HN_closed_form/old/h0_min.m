%minimizer
clearvars;close all;clc;
Maturity = 30:30:210;
S = 1;
K = 0.9:0.05:1.20;
K = K*S;
r = 0.005;
prices_opt =-ones(1,length(K));
while any(any(prices_opt < 0)) || any(any(prices_opt > 2*(max(K)-S)))
    a = 1;b = 1;g = 1;
    while (b+a*g^2 >= 1) || (b+a*g^2 <= 0.8)
        a = 1.0e-6 + (1.5e-6-1.0e-6).*rand(1,1);
        b = .57 + (.70-.57).*rand(1,1);
        g = 450 + (500-450).*rand(1,1);
        disp(b+a*g^2);
    end
    w = 5.5e-7 + (1e-6-5.5e-7).*rand(1,1);
    sig0 =5*(w+a)/(1-a*g^2-b);
    args=fmincon(@(sig) error1(sig,K,S,Maturity,r,w,a,b,g),sig0,[],[],[],[],0.000001,0.001);
    prices_opt = zeros(length(Maturity),length(K));
    for k = 1:length(K)
        for t=1:length(Maturity)
            prices_opt(t,k)= HestonNandi(S,K(k),args,Maturity(t),r/365,w,a,b,g,0);
        end
    end
    disp(min(min(prices_opt)));
end
for k = 1:length(K)
    for t=1:length(Maturity)
        iv(t,k) = blsimpv(S,K(k),r,Maturity(t)/252,prices_opt(t,k),10,0,1e-15,true);
    end
end
figure('Name','Prices and Vola')
[X,Y]=meshgrid(K,Maturity);
subplot(1,2,1)
surf(X,Y,prices_opt)
subplot(1,2,2)
surf(X,Y,iv)

% %%
% prices_bs = zeros(1,length(K));
% prices_hng = zeros(1,length(K));
% iv = zeros(1,length(K));
% for k = 1:length(K)
%     prices_hng(k)= HestonNandi(S,K(k),(w+a)/(1-a*g^2-b),Maturity,r/365,w,a,b,g,0);
%     prices_bs(k)= blsprice(S,K(k),r,Maturity/252,sqrt(252*(w+a)/(1-a*g^2-b)));
% end
% for k = 1:length(K)%min(length(K(prices3>0)),length(K(prices1>0)))
%     iv(k) = blsimpv(S,K(k),r,Maturity/252,prices_opt(k),10,0,1e-15,true);
% end
% figure
% plot(K,prices_opt);hold on,plot(K,prices_bs);hold on,plot(K,prices_hng);
% set(gca,'YScale','log')
% legend('optHNG','BS','HNG')
% 
% figure
% plot(K,iv)
% legend('optHNG')
%%
function y = error1(Sig_,K,S,Maturity,r,w,a,b,g)
        for k = 1:length(K)
            for t=1:length(Maturity)
                prices_hn(t,k)= HestonNandi(S,K(k),Sig_,Maturity(t),r/365,w,a,b,g,0);
                prices_bs(t,k)=blsprice(S,K(k),r,Maturity(t)/252,sqrt(252*Sig_));%(w+a)/(1-a*g^2-b)));
            end
        end
        y=sum(sum(abs(prices_bs-prices_hn).^2.));%/abs(prices_bs)));
end
