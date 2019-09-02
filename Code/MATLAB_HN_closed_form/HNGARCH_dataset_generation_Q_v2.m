clearvars -except params_risk_neutral params_risk_neutral_no
%Maturity = [30, 60, 90, 120, 150];%, 180, 210, 240];%, 270, 300, 330, 360];
Maturity = 40:20:120;
%K = [0.8, 0.84, 0.89, 0.93, 0.98, 1.02, 1.07, 1.11, 1.16, 1.2];
K = 0.90:0.03:1.18;
Nmaturities = length(Maturity);
Nstrikes = length(K);
S = 1;
r = 0.05/252;
j = 0;
l = 0;
Nsim = 1 ;
scenario_data = zeros(Nsim, 7+Nstrikes*Nmaturities);
fprintf('%s','Generatiting Prices. Progress: 0%')
for i = 1:Nsim
    price = -ones(Nmaturities,Nstrikes);
    while any(any(price < 0)) || any(any(price > 0.45))
        a = 1;
        b = 1;
        g = 1;
        while (b+a*g^2 >= 1)
            a = 6e-7 + (2e-6-6e-7).*rand(1,1);
            b = .4 + (.6-.4).*rand(1,1);
            g = 450 + (500-450).*rand(1,1);
            % 95% quantil mit h(0) optimierung
            %a = 1.08e-8 + (2.35e-6-1.08e-8).*rand(1,1);
            %b = .43 + (.97-.43).*rand(1,1);
            %g = 453 + (477-453).*rand(1,1);
            % 95% quantil ohne h(0) optimierung
            %a = 5.8e-7 + (1.4e-6-5.8e-7).*rand(1,1);
            %b = .43 + (.75-.43).*rand(1,1);
            %g = 441 + (590-441).*rand(1,1);
            %a=0.01;
            %b=0.5;
            %g=5;
            disp(a*g^2+b)
        end
        w = 5e-7 + (1.2e-6-5e-7).*rand(1,1);
        %Sig_ = 1e-7 + (1e-3-1e-7).*rand(1,1);
        % 95% quantil mit h(0) optimierung
        %w = 1.6e-6 + (3.2e-6-1.6e-6).*rand(1,1);
        %Sig_ = 4.5e-5 + (1e-3-4.5e-5).*rand(1,1);
        % 95% quantil ohne h(0) optimierung
        %w = 4.1e-7 + (2.9e-6-4.1e-7).*rand(1,1);
        var = 0.0;
        Sig_ = (w+a)/(1-b-a*g^2)*(1-var+2*var*rand(1,1));
        %Sig_ = w+(a*g^2+b)*(w+a)/(1-b-a*g^2)*(1-var+2*var*rand(1,1));
        lam = 0.0;
        for t = 1:Nmaturities
            Sig_=Sig_*(Maturity(t)/mean(Maturity));
            for k = 1:Nstrikes
                Sig_ = Sig_*(K(k)/mean(K));
                price(t,k)= HestonNandi(S,K(k),Sig_,Maturity(t),r,w,a,b,g,lam);
            end
        end
        disp([min(min(price)),max(max(price))])
    end
    if ismember(i,floor(Nsim*[1/(10*log10(Nsim*100)):1/(10*log10(Nsim*100)):1]))
        fprintf('%0.5g',i/Nsim*100),fprintf('%s',"%")
    end
    scenario_data(i,:) = [a, b, g, w, Sig_,(a+w)/(1-a*g^2-b), b+a*g^2, reshape(price', [1,Nstrikes*Nmaturities])];  
end
fprintf('\n')
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
%

prices_all = scenario_data(:, 8:end);
iv = zeros(size(prices_all));
iv_p = zeros(Nstrikes,Nmaturities);
fprintf('%s','ImpVola Progress: 0%')
for i = 1:Nsim
    if ismember(i,floor(Nsim*[1/(10*log10(Nsim*100)):1/(10*log10(Nsim*100)):1]))
        fprintf('%0.5g',i/Nsim*100),fprintf('%s',"%")
    end
    price_new = reshape(prices_all(i,:), [Nstrikes, Nmaturities]);
    for t = 1:Nmaturities
        for k = 1:Nstrikes
            iv_p(k,t) = blsimpv(S,K(k),r*252,Maturity(t)/252,price_new(k,t));
        end
    end
iv(i,:) =  reshape(iv_p, [1,Nstrikes*Nmaturities]);
end
fprintf('\n')
iv1 = iv(~any(isnan(iv),2),:);
disp(strcat("rate/number of nonzeros vola-data: ",num2str(nnz(iv1)/(Nsim*Nstrikes*Nmaturities)*100),"% , ",num2str(nnz(iv1))));
disp(['max vola: ', num2str(max(max(iv1)))])
disp(['min vola: ', num2str(min(min(iv1)))])
disp(['mean vola: ', num2str(mean(mean(iv1)))])
disp(['median vola: ', num2str(median(median(iv1)))])
disp(['low volas: ', num2str(length(iv1(iv1<.07)))])
data_price = scenario_data(:,[1:5,8:end]);
data_vola = [data_price(:,[1:5]),iv];
data_vola = data_vola(~any(isnan(data_vola),2),:);
%%
figure, surf(reshape(data_vola(1,6:end),[13,12]))%,zlim([0 1])
%save(strcat('data_price_',num2str(length(data_price)),'_',num2str(r*252),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity))),'data_price')
%save(strcat('data_vola_',num2str(length(data_vola)),'_',num2str(r*252),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity))),'data_vola')
%% 
%figure
%boxplot(reshape(iv1,[1,Nsim*156]))
%figure
%histogram(reshape(iv1,[1,4137*156]))
figure
[X,Y] = meshgrid(Maturity,K);
surf(X,Y,iv_p);
figure
subplot(2,3,1)
histogram(iv1(:,1))
subplot(2,3,2)
histogram(iv1(:,2))
subplot(2,3,3)
histogram(iv1(:,10))
subplot(2,3,4)
histogram(iv1(:,30))
subplot(2,3,5)
histogram(iv1(:,80))
subplot(2,3,6)
histogram(iv1(:,end))

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