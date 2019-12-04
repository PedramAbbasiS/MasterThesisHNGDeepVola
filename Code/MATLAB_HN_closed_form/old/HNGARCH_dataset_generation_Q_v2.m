clearvars -except params_risk_neutral params_risk_neutral_no
close all;
%rng('default')
%Maturity = [30, 60, 90, 120, 150];%, 180, 210, 240];%, 270, 300, 330, 360];
Maturity = 30:30:240;
%K = 0.9:0.025:1.1;
S = 1;
K = 0.9:0.02:1.1;
K = K*S;
Nmaturities = length(Maturity);
Nstrikes = length(K);
r = 0.005;
Nsim = 25000 ;
scenario_data = zeros(Nsim, 7+Nstrikes*Nmaturities);
fprintf('%s','Generatiting Prices. Progress: 0%')
for i = 1:Nsim
    price = -ones(Nmaturities,Nstrikes);
    while any(any(price < 0)) || any(any(price > 1.5*(S-min(K))))
        a = 1;
        b = 1;
        g = 1;
        while (b+a*g^2 >= 1)||(b+a*g^2 <= 0.75)
            %%a = 1.0e-7 + (1.5e-6-1.0e-7).*rand(1,1);
            %a = 1.0e-6 + (1.5e-4-1.0e-6).*rand(1,1);
            %a= 1.51e-6;
            %a= 1.5e-6+rand(1,1)*(0.02e-6);
            %b = .57 + (.70-.57).*rand(1,1);
            %b=0.662;
            %b=0.66+rand(1,1)*0.004;
            %g =sqrt((rand(1,1)*(0.999-0.75)+0.75-b)/a);%
            %g=450 + (500-450).*rand(1,1);
            %g=480 + (550-480).*rand(1,1);
            %b= rand(1,1)*(0.999999-0.7)+0.75-a*g^2;
            %g = 464;
            %g =464+rand(1,1);
            % 95% quantil mit h(0) optimierung
            %a = 1.08e-8 + (2.35e-6-1.08e-8).*rand(1,1);
            %b = .43 + (.97-.43).*rand(1,1);
            %g = 453 + (477-453).*rand(1,1);
            % 95% quantil ohne h(0) optimierung
            a = 1.0e-6 + (1.5e-6-1.0e-6).*rand(1,1);
            b = .57 + (.70-.57).*rand(1,1);
            g = 450 + (500-450).*rand(1,1);
            %disp(a*g^2+b)
        end
        %w = (15+(20-15)*rand(1,1))*(5.5e-7 + (1e-6-5.5e-7).*rand(1,1));
        %w = 4.29e-7;
        %w = 4.2e-7+rand(1,1)*(0.2e-7);
        %Sig_ = 1e-7 + (1e-3-1e-7).*rand(1,1);
        % 95% quantil mit h(0) optimierung
        %w = 1.6e-6 + (3.2e-6-1.6e-6).*rand(1,1);
        %Sig_ = 4.5e-5 + (1e-3-4.5e-5).*rand(1,1);
        % 95% quantil ohne h(0) optimierung
        
        w = 20*(5.5e-7 + (1e-6-5.5e-7).*rand(1,1));
        Sig_ = (w+a)/(1-a*g^2-b);
        lam = 0.0;
        for t = 1:Nmaturities
            for k = Nstrikes:-1:1
                price(t,k)= HestonNandi(S,K(k),Sig_,Maturity(t),r/365,w,a,b,g,lam);
                if any(any(price < 0)) || any(any(price > 1.5*(S-min(K))))
                    continue
                end
            end
            if any(any(price < 0)) || any(any(price > 1.5*(S-min(K))))
                continue
            end
        end
        %disp([min(min(price)),max(max(price))])
        if any(any(price < 0)) || any(any(price > 1.5*(S-min(K))))
            continue
        end
        
    end
    if ismember(i,floor(Nsim*[1/(5*log10(Nsim*100)):1/(5*log10(Nsim*100)):1]))
        fprintf('%0.5g',round(i/Nsim*100,1)),fprintf('%s',"%")
    end
    scenario_data(i,:) = [a, b, g, w, Sig_,(a+w)/(1-a*g^2-b), b+a*g^2, reshape(price', [1,Nstrikes*Nmaturities])];  
end
fprintf('\n')
for i=1
    disp(strcat("rate/number of nonzeros price-data: ",num2str(nnz(scenario_data(:,8:end))/(Nsim*Nstrikes*Nmaturities)*100),"% , ",num2str(nnz(scenario_data(:,8:end)))));
    disp(['max price: ', num2str(max(max(scenario_data(:,8:end))))])
    disp(['min price: ', num2str(min(min(scenario_data(:,8:end))))])
    disp(['mean price: ', num2str(mean(mean(scenario_data(:,8:end))))])
    disp(['median price: ', num2str(median(median(scenario_data(:,8:end))))])
    disp(['median alpha: ', num2str(median(scenario_data(:,1)))])
    disp(['median beta: ', num2str(median(scenario_data(:,2)))])
    disp(['median gamma: ', num2str(median(scenario_data(:,3)))])
    disp(['median omega: ', num2str(median(scenario_data(:,4)))])
    disp(['median sigma: ', num2str(median(scenario_data(:,5)))])
    disp(['median stationary constraint: ', num2str(median(scenario_data(:,6)))])
end

prices_all = scenario_data(:, 8:end);
iv = zeros(size(prices_all));
iv_p = zeros(Nstrikes,Nmaturities);
fprintf('%s','ImpVola Progress: 0%')
for i = 1:Nsim
    price_new = reshape(prices_all(i,:), [Nstrikes, Nmaturities]);
    for t = 1:Nmaturities
        for k = 1:Nstrikes
            iv_p(k,t) = blsimpv(S,K(k),r,Maturity(t)/252,price_new(k,t),10,0,1e-8,true);
        end
    end
    if ismember(i,floor(Nsim*[1/(5*log10(Nsim*100)):1/(5*log10(Nsim*100)):1]))
        fprintf('%0.5g',round(i/Nsim*100,1)),fprintf('%s',"%")
    end
    iv(i,:) =  reshape(iv_p, [1,Nstrikes*Nmaturities]);
end
fprintf('\n')
iv1 = iv(~any(isnan(iv),2),:);

for i=1
    disp(strcat("rate/number of nonzeros vola-data: ",num2str(nnz(iv1)/(Nsim*Nstrikes*Nmaturities)*100),"% , ",num2str(nnz(iv1))));
    disp(['max vola: ', num2str(max(max(iv1)))])
    disp(['min vola: ', num2str(min(min(iv1)))])
    disp(['mean vola: ', num2str(mean(mean(iv1)))])
    disp(['median vola: ', num2str(median(median(iv1)))])
    disp(['low volas: ', num2str(length(iv1(iv1<.07)))])
    data_price = scenario_data(:,[1:5,8:end]);
    data_vola = [data_price(:,1:5),iv];
    data_vola_clear = data_vola(~any(isnan(data_vola),2),:);
    data_price_clear = data_price(~any(isnan(data_vola),2),:);  
    yearly_h0 = mean(sqrt(252*data_vola_clear(:,5)));
end
%
figure
[X,Y] = meshgrid(Maturity,K);
for i = 1:8
    surf(X,Y,reshape(data_vola_clear(i,6:end),size(price_new)));hold on
end

%zlim([0,0.5])
save(strcat('data_price_',num2str(length(data_price_clear)),'_',num2str(r),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity))),'data_price_clear')
save(strcat('data_vola_',num2str(length(data_vola_clear)),'_',num2str(r),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity))),'data_vola_clear')
%% 
diff = zeros(Nmaturities,Nstrikes,size(data_price_clear,1));
pricer = zeros(Nmaturities,Nstrikes,size(data_price_clear,1));
for t = 1:Nmaturities
    for k =1:Nstrikes
        for i=1:size(data_price_clear,1)
            pricer(t,k,i)=blsprice(1,K(k),r,Maturity(t)/252,sqrt(252*data_price_clear(i,5)));%(w+a)/(1-a*g^2-b))); 
            tmp = reshape(data_price_clear(i,6:end),[Nmaturities,Nstrikes]);
            diff(t,k,i) = abs(pricer(t,k,i)-tmp(t,k))/pricer(t,k,i)*100;
        end
    end
end
mean_diff = mean(diff,3);
median_diff = median(diff,3);
median_total = median(median(median(diff)));
mean_total = mean(mean(mean(diff)));
params = data_vola_clear(:,1:5);sig=(params(:,1)+params(:,4))./(1-params(:,1).*params(:,3).^2-params(:,2));quote=params(:,5)./sig;
figure, histogram(quote);
figure, for i=1:size(data_vola_clear,1), surf(diff(:,:,i));hold on,set(gca,'ZScale','log','ColorScale','log'),end

figure
[X,Y] =meshgrid(K,Maturity);
subplot(1,2,1)
    title('Mean Relative Error')
    surf(X,Y,mean_diff)
    zlabel('Error in Percent')
    set(gca,'ColorScale','log')
subplot(1,2,2)
    title('Median Relative Error')
    surf(X,Y,median_diff)
    zlabel('Error in Percent')
    set(gca,'ColorScale','log')
    %surf(X,Y,pricer);hold on
%surf(X,Y,price);
%legend('BS','HNG')
%set(gca, 'ZScale', 'log')
%figure
%boxplot(reshape(iv1,[1,Nsim*156]))
%figure
%histogram(reshape(iv1,[1,4137*156]))
% 
% figure
% subplot(2,3,1)
% histogram(iv1(:,1))
% subplot(2,3,2)
% histogram(iv1(:,2))
% subplot(2,3,3)
% histogram(iv1(:,10))
% subplot(2,3,4)
% histogram(iv1(:,30))
% subplot(2,3,5)
% histogram(iv1(:,80))
% subplot(2,3,6)
% histogram(iv1(:,end))

%plot(,iv_p)
% %%
% for i = 1:length(iv1)
%     diff(i) = max(iv1(i,:))-min(iv1(i,:));
% end
% 
% [size1, size2] = size(iv1);
% iv_re = reshape(iv1, [1, size1*size2]);
% ksdensity(iv_re)
% 
% 
% data = [scenario_data(:,1:6),iv];
% data = data(~any(isnan(data),2),:);
% save('data_v2_10000.mat', 'data')
% 
% %%
% re2 = zeros(Nsim,Nmaturities*Nstrikes);
% for i = 1:length(scenario_data)
%     re1 = reshape(scenario_data(i,7:end), [Nmaturities, Nstrikes])';
%     re2(i,:) = reshape(re1, [1, Nmaturities*Nstrikes]);
% end
% scenario_data_2 = [scenario_data(:,1:6),re2];