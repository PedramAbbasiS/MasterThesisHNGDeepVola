clearvars
Maturity = [30, 60, 90, 120, 150, 180, 210, 240];
K = 0.9:0.02:1.1; 
Nmaturities = length(Maturity);
Nstrikes = length(K);
S = 1;
r = .005/252;
Nsim = 21000;
scenario_data = zeros(Nsim, 6+Nstrikes*Nmaturities);
for i = 1:Nsim
    if ismember(i,floor(Nsim*(0.025:0.025:1)))
        disp(strcat(num2str(i/Nsim*100),"%"))
    end
    price = -ones(Nmaturities,Nstrikes);
    while any(any(price < 0)) || any(any(price > 1.5*(S-min(K))))
        a = 1;
        b = 1;
        g = 1;
        while (b+a*g^2 >= 1) || (b+a*g^2 <= 0.85)
            %a = (5e-6 + (50*1.5e-6-5e-6).*rand(1,1));
            %b = (.851 + (.98-.851).*rand(1,1));
            %g = (1 + (4-1).*rand(1,1));
            a = (5.8e-7 + (1.4e-6-5.8e-7).*rand(1,1));
            b = (.43 + (.75-.43).*rand(1,1));
            g = (441 + (590-441).*rand(1,1));
        end
        %w = (5*5.5e-7 + (15e-6-5*5.5e-7).*rand(1,1));
        w = 20*(4.1e-7 + (2.9e-6-4.1e-7).*rand(1,1));
        Sig_ = (w+a)/(1-b-a*g^2);
        lam = 0.0;
        for t = 1:Nmaturities
            for k = Nstrikes:-1:1
                price(t,k)= HestonNandi(S,K(k),Sig_,Maturity(t),r,w,a,b,g,lam);
                if any(any(price < 0)) || any(any(price > 1.5*(S-min(K))))
                    continue
                end
            end
            if any(any(price < 0)) || any(any(price > 1.5*(S-min(K))))
                continue
            end
        end
        if any(any(price < 0)) || any(any(price > 1.5*(S-min(K))))
            continue
        end
    end
    scenario_data(i,:) = [a, b, g, w, Sig_, b+a*g^2, reshape(price', [1,Nstrikes*Nmaturities])];  
end                              
disp(['number of nonzeros price-data: ', num2str(nnz(scenario_data))])
disp(['max price: ', num2str(max(max(scenario_data(:,7:end))))])
disp(['min price: ', num2str(min(min(scenario_data(:,7:end))))])
disp(['mean price: ', num2str(mean(mean(scenario_data(:,7:end))))])
disp(['median price: ', num2str(median(median(scenario_data(:,7:end))))])

disp(['median alpha: ', num2str(median(scenario_data(:,1)))])
disp(['median beta: ', num2str(median(scenario_data(:,2)))])
disp(['median gamma: ', num2str(median(scenario_data(:,3)))])
disp(['median omega: ', num2str(median(scenario_data(:,4)))])
disp(['median sigma: ', num2str(median(scenario_data(:,5)))])
disp(['median stationary constraint: ', num2str(median(scenario_data(:,6)))])
disp(['median long-term vola: ', num2str(median(sqrt(252*scenario_data(:,5))))])

[X,Y] = meshgrid(K,Maturity);
surf(X,Y,reshape(scenario_data(1,7:end),[Nstrikes, Nmaturities])')
%%
prices_all = scenario_data(:, 7:end);
iv = zeros(size(scenario_data(:,7:end)));
iv_p = zeros(Nstrikes,Nmaturities);
for i = 1:Nsim
    price_new = reshape(prices_all(i,:), [Nstrikes, Nmaturities]);
    for t = 1:Nmaturities
        for k = 1:Nstrikes
            iv_p(k,t) = blsimpv(S,K(k),r*252,Maturity(t)/252, price_new(k,t));
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

plot(K,iv1(1,1:length(K)))
[X,Y] = meshgrid(K,Maturity);
surf(X,Y,reshape(iv1(2,:),[Nstrikes, Nmaturities])')

%%
%check with black-scholes:
[valid_sim,~]= size(iv1);
price_bs = zeros(Nmaturities,Nstrikes);
price_bs_all = zeros(valid_sim,Nstrikes*Nmaturities);

data_hn_new = [scenario_data(:,1:6),iv];
data_hn_new = data_hn_new(~any(isnan(data_hn_new),2),:);
%price_hn_new = price_hn_new(:,1:Nstrikes*Nmaturities);

for i = 1:valid_sim
    for t = 1:Nmaturities
        for k = 1:Nstrikes 
            price_bs(t,k) = blsprice(S,K(k),r*252,Maturity(t)/252,sqrt(252*data_hn_new(i,5)));
        end
    end
    price_bs_all(i,:) = reshape(price_bs', [1,Nstrikes*Nmaturities]);
end
price_hn_new = [scenario_data(:,7:end),iv];
price_hn_new = price_hn_new(~any(isnan(price_hn_new),2),:);
price_hn_new = price_hn_new(:,1:Nstrikes*Nmaturities);
diff_bs_hn = abs(price_bs_all - price_hn_new)./abs(price_bs_all);
disp(['mean error regarding BS: ', num2str(mean(mean(diff_bs_hn)))])
disp(['median error regarding BS: ', num2str(median(median(diff_bs_hn)))])
disp(['maximal error regarding BS: ', num2str(max(max(diff_bs_hn)))])
%%
[size1, size2] = size(iv1);
iv_re = reshape(iv1, [1, size1*size2]);
ksdensity(iv_re)
ksdensity(iv1(:,4))

%zuschneiden!!!!!
data = [scenario_data(:,1:6),iv];
data = data(~any(isnan(data),2),:);
%data = data(1:20000,:);
save('data_vola_w20_1000_08_12_0025.mat', 'data')


data = [scenario_data,iv];
data = data(~any(isnan(data),2),:);
data = data(:,1:6+Nstrikes*Nmaturities);
%data = data(1:20000,:);
save('data_price_w20_1000_08_12_0025.mat', 'data')


%re2 = zeros(Nsim,Nmaturities*Nstrikes);
%for i = 1:length(scenario_data)
%    re1 = reshape(scenario_data(i,7:end), [Nmaturities, Nstrikes])';
%    re2(i,:) = reshape(re1, [1, Nmaturities*Nstrikes]);
%end
%scenario_data_2 = [scenario_data(:,1:6),re2];



%plots
min_ = min(iv1);
max_ = max(iv1);
mean_ = mean(iv1);
median_ = median(iv1);

min_re = reshape(min_,[Nstrikes,Nmaturities]);
max_re = reshape(max_,[Nstrikes,Nmaturities]);
mean_re =reshape(mean_,[Nstrikes,Nmaturities]);
median_re = reshape(median_,[Nstrikes,Nmaturities]);
[X,Y] = meshgrid(K,Maturity);
surf(X,Y,min_re')
hold on
surf(X,Y,max_re')
hold on
surf(X,Y,mean_re')
hold on
surf(X,Y,median_re')


mean_2 = median(iv1(1:end-1,:),2);
id_min = find(mean_2 == min(mean_2));
id_max = find(mean_2 == max(mean_2));
id_median = find(mean_2 == median(mean_2));
min_re_2 = reshape(iv1(id_min,:),[Nstrikes,Nmaturities]);
max_re_2 = reshape(iv1(id_max,:),[Nstrikes,Nmaturities]);
median_re_2 = reshape(iv1(id_median,:),[Nstrikes,Nmaturities]);
[X,Y] = meshgrid(K,Maturity);
surf(X,Y,min_re_2')
hold on
surf(X,Y,max_re_2')
hold on
surf(X,Y,median_re_2')

