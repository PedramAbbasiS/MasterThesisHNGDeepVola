close all
clearvars -except params_risk_neutral params_risk_neutral_no
Maturity = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360];
%Maturity = 120;
%K = [0.8, 0.84, 0.89, 0.93, 0.98, 1.02, 1.07, 1.11, 1.16, 1.2];
K = 0.9:0.05:1.1;
Nmaturities = length(Maturity);
Nstrikes = length(K);
S = 1;
r = 0.025/252;
j = 0;
l = 0;
Nsim = 2 ;
scenario_data = zeros(Nsim, 7+Nstrikes*Nmaturities);
sig_mat = zeros(Nsim,1);
sig_mat_rn = zeros(Nsim,1);
price = zeros(Nmaturities,Nstrikes);
a = 1e-7; %alpha 40% änderung führrt zu 1e-3 vola/price änderung
b = .6; % 12% änderung führr zu 1e-3 price und 1e-2 vola änderung
g = 450; % 9% zu 1e-1 änderung
w = [5e-6,1.22*5e-6]; % 22% zu 1e-3 in price ,1e-2 in vola
var = 0.0;
Sig_ = (w+a)./(1-b-a*g.^2)*(1-var+2*var*rand(1,1));
lam = 0.0;
for i = 1:2
    for t = 1:Nmaturities
        for k = 1:Nstrikes
            price(t,k)= HestonNandi(S,K(k),Sig_(i),Maturity(t),r,w(i),a,b,g,lam);
        end
    end
    scenario_data(i,:) = [a, b, g, w(i), Sig_(i),(a+w(i))/(1-a*g^2-b), b+a*g^2, reshape(price', [1,Nstrikes*Nmaturities])];  
end

prices_all = scenario_data(:, 8:end);
figure 
[X,Y] = meshgrid(Maturity,K);
surf(X,Y,reshape(prices_all(2,:),Nmaturities,Nstrikes)'-reshape(prices_all(1,:),Nmaturities,Nstrikes)');% hold on;

iv = zeros(size(scenario_data(:,8:end)));
iv_p = zeros(Nstrikes,Nmaturities);
for i = 1:Nsim
    price_new = reshape(prices_all(i,:), [Nstrikes, Nmaturities]);
    for t = 1:Nmaturities
        for k = 1:Nstrikes
            iv_p(k,t) = blsimpv(S,K(k),r,Maturity(t)/252,price_new(k,t));
        end
    end
iv(i,:) =  reshape(iv_p, [1,Nstrikes*Nmaturities]);
end
iv1 = iv(~any(isnan(iv),2),:);
%
figure
[X,Y] = meshgrid(Maturity,K);
surf(X,Y,iv_p-reshape(iv1(1,:),Nstrikes,Nmaturities));% hold on;
%surf(X,Y,reshape(iv1(2,:),Nstrikes,Nmaturities));
