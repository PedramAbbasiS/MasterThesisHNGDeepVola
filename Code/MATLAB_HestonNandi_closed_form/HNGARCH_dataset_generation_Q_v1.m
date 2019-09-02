clearvars
Maturity = [30, 60, 90, 120, 150, 180, 210, 240];
K = 0.7:0.05:1.3;
Nmaturities = length(Maturity);
Nstrikes = length(K);
S = 1;
r = 0.025/252;
j = 0;
l = 0;
Sig_=.04/252;
Nsim = 2000;
scenario_data = zeros(Nsim, 6+Nstrikes*Nmaturities);
for i = 1:Nsim
    %disp(i)
    %j = j+1;
    if ismember(i,floor(Nsim*[0.025:0.025:1]))
        disp(strcat(num2str(i/Nsim*100),"%"))
    end
    %if (j == 250)
    %    l = l + 1;
    %    disp(l*250)
    %    j = 0; 
    %end
    price = -ones(Nmaturities,Nstrikes);
    while any(any(price < 0)) || any(any(price > 0.5))
        a = 1;
        b = 1;
        g = 1;
        while (b+a*g^2 > 1)
            a = 1e-8 + (1e-6-1e-8).*rand(1,1);
            b = .5 + (.65-.5).*rand(1,1);
            g = 400 + (600-400).*rand(1,1);
            % 95% quantil mit h(0) optimierung
            %a = 1.08e-8 + (2.35e-6-1.08e-8).*rand(1,1);
            %b = .43 + (.97-.43).*rand(1,1);
            %g = 453 + (477-453).*rand(1,1);
            % 95% quantil ohne h(0) optimierung
            %a = 5.8e-7 + (1.4e-6-5.8e-7).*rand(1,1);
            %b = .43 + (.75-.43).*rand(1,1);
            %g = 441 + (590-441).*rand(1,1);
        end
        w = 7.55e-6 + (3.45e-4-7.55e-6).*rand(1,1);
        Sig_ = (w+a)/(1-a*g^2-b);
        %1e-4 + (1e-3-1e-4).*rand(1,1);
        % 95% quantil mit h(0) optimierung
        %w = 1.6e-6 + (3.2e-6-1.6e-6).*rand(1,1);
        %Sig_ = 4.5e-5 + (1e-3-4.5e-5).*rand(1,1);
        % 95% quantil ohne h(0) optimierung
        %w = 4.1e-7 + (2.9e-6-4.1e-7).*rand(1,1);
        %Sig_ = (w+a)/(1-b-a*g^2);
        lam = 0.0;
        for t = 1:Nmaturities
            for k = 1:Nstrikes
                price(t,k)= HestonNandi(S,K(k),Sig_,Maturity(t),r,w,a,b,g,lam);
            end
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
%%

prices_all = scenario_data(:, 7:end);
iv = zeros(size(scenario_data(:,7:end)));
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
disp(['number of nonzeros vola-data: ', num2str(nnz(iv1))])
disp(['max vola: ', num2str(max(max(iv1)))])
disp(['min vola: ', num2str(min(min(iv1)))])
disp(['mean vola: ', num2str(mean(mean(iv1)))])
disp(['median vola: ', num2str(median(median(iv1)))])
disp(['low volas: ', length(iv1(iv1<.07))])

plot(K,iv1(2,1:13))

%%
for i = 1:length(iv1)
    diff(i) = max(iv1(i,:))-min(iv1(i,:));
end

[size1, size2] = size(iv1);
iv_re = reshape(iv1, [1, size1*size2]);
ksdensity(iv_re)
ksdensity(iv1(:,4))


data = [scenario_data(:,1:6),iv];
data = data(~any(isnan(data),2),:);
save('data_v2_2000_new.mat', 'data')


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

