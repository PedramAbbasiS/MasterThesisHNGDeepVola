clearvars
Maturity = [30, 60, 90, 120, 150, 180, 210, 240];
K = [0.8, 0.84, 0.89, 0.93, 0.98, 1.02, 1.07, 1.11, 1.16, 1.2];
Nmaturities = length(Maturity);
Nstrikes = length(K);
S = 1;
r = 0.0;
Sig_=.04/252;
Nsim = 100; %2100;
scenario_data = zeros(Nsim, 7+Nstrikes*Nmaturities);
for i = 1:Nsim
    disp(i)
    price = -ones(Nmaturities,Nstrikes);
    while any(any(price < 0)) || any(any(price > 0.45))
        a = 1;
        b = 1;
        g = 1;
        while (b+a*g^2 > 1)
            a = 1e-6 + (9e-6-1e-6).*rand(1,1);
            b = .6 + (.7-.6).*rand(1,1);
            g = 400 + (550-400).*rand(1,1);
        end
        w = 3.5e-7 + (5.2e-7-3.5e-7).*rand(1,1);
        Sig_ = (w+a)/(1-a*g^2-b);
        lam = .2 + (1.1-.2).*rand(1,1);
        for t = 1:Nmaturities
            for k = 1:Nstrikes
                price(t,k)= HestonNandi_P(S,K(k),Sig_,Maturity(t),r,w,a,b,g,lam);
            end
        end
        disp(['max price: ', num2str(max(max(price)))])
        disp(['min price: ', num2str(min(min(price)))])
    end
    scenario_data(i,:) = [a, b, g, w, Sig_, lam, b+a*g^2, reshape(price, [1,Nstrikes*Nmaturities])];  
end                              
disp(['number of nonzeros price-data: ', num2str(nnz(scenario_data))])
disp(['max price: ', num2str(max(max(scenario_data(:,8:end))))])
disp(['min price: ', num2str(min(min(scenario_data(:,8:end))))])
disp(['mean price: ', num2str(mean(mean(scenario_data(:,8:end))))])
disp(['median price: ', num2str(median(median(scenario_data(:,8:end))))])

disp(['median alpha: ', num2str(median(scenario_data(:,1)))])
disp(['median beta: ', num2str(median(scenario_data(:,2)))])
disp(['median gamma: ', num2str(median(scenario_data(:,3)))])
disp(['median omega: ', num2str(median(scenario_data(:,4)))])
disp(['median sigma: ', num2str(median(scenario_data(:,5)))])
disp(['median lambda: ', num2str(median(scenario_data(:,6)))])
disp(['median stationary constraint: ', num2str(median(scenario_data(:,7)))])


prices_all = scenario_data(:,8:end);
iv = zeros(size(scenario_data(:,8:end)));
iv_p = zeros(Nmaturities, Nstrikes);
for i = 1:Nsim
    price_new = reshape(prices_all(i,:), [Nmaturities, Nstrikes]);
    for t = 1:Nmaturities
        for k = 1:Nstrikes
            iv_p(t,k) = blsimpv(S,K(k),r,Maturity(t)/252,price_new(t,k));
        end
    end
iv(i,:) =  reshape(iv_p, [1,Nstrikes*Nmaturities]);
end
iv1 = iv(~any(isnan(iv),2),:);
disp(['number of nonzeros vola-data: ', num2str(nnz(iv1))])
disp(['max vola: ', num2str(max(max(iv1)))])
disp(['min vola: ', num2str(min(min(iv1)))])
disp(['mean vola: ', num2str(mean(mean(iv1)))])
disp(['median vola: ', num2str(median(median(iv1)))])
disp(['low volas: ', length(iv1(iv1<.07))])

[size1, size2] = size(iv1);
iv_re = reshape(iv1, [1, size1*size2]);
ksdensity(iv_re)


data = [scenario_data(:,1:7),iv];
data = data(~any(isnan(data),2),:);
save('data_P_v2.mat', 'data')