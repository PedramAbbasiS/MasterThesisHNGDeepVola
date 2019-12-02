clearvars; clc;close all;
load('/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Code/MATLAB_HN_MLE/generaldata2015.mat')
load('/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Code/MATLAB_HN_MLE/params_Options_2015_MRAEfull.mat')
load('/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Code/MATLAB_HN_MLE/weekly_2015_mle.mat')
%dimension 
%for i = 48:53
%   values{i} = struct();
%end
for i=1:47
    compare(i,1,:) = params_Q_mle_weekly(i,:);
    compare(i,2,:) = values{i}.hngparams;
end

compare2 = permute(compare,[2,3,1]);
%
%rng('default')
N               = length(values);
Maturity        = 30:30:210;
K               = 0.9:0.025:1.1;
S               = 1;
K               = K*S;
Nmaturities     = length(Maturity);
Nstrikes        = length(K);
r               = 0.005; %yearly
Nsim            = 5;%100;%5000;
data_vec        = [combvec(K,Maturity);S*ones(1,Nmaturities*Nstrikes)]';

%% Version with szenario generation based on random parameter choice
scenario_data = zeros(2*Nsim*N, 7+Nstrikes*Nmaturities);
idx_vec(:,1) = randi(N,2*N*Nsim,1);
idx_vec(:,2) = randi(2,2*N*Nsim,1);
std_vec      = std(compare); 
fprintf('%s','Generatiting Prices. Progress: 0%')
fails = -2*2*N*Nsim;
for i = 1:2*N*Nsim
    params = reshape(compare(idx_vec(i,1),idx_vec(i,2),:),1,4);
    price  = -ones(1,Nmaturities*Nstrikes);
    while any(any(price < 0)) %|| any(any(price > 1.5*(S-min(K))))
        a = 1;
        b = 1;
        g = 1;
        w = 0;
        % Optimize random draw to fit distribution better!!!
        while (b+a*g^2 >= 1) || a<0  || b<0
            a = params(2) + std_vec(1,idx_vec(i,2),2)*randn(1,1);
            b = params(3) + std_vec(1,idx_vec(i,2),3)*randn(1,1);
            g = params(4) + std_vec(1,idx_vec(i,2),4)*randn(1,1);
            fails = fails +1;
        end
        while w<=0
            w = params(1) + std_vec(1,idx_vec(i,2),1)*randn(1,1);
            fails = fails +1;
        end
        %Sig_  = (w+a)/(1-a*g^2-b);
        Sig_  = sig2_0(idx_vec(i,1))*(0.9+0.2*rand(1,1));
        price = price_Q_clear([w,a,b,g],data_vec,r/252,Sig_);
    end
    if ismember(i,floor(2*Nsim*N*[1/(5*log10(2*Nsim*N*100)):1/(5*log10(2*Nsim*N*100)):1]))
        fprintf('%0.5g',round(i/(2*N*Nsim)*100,1)),fprintf('%s',"%")
    end
    scenario_data(i,:) = [a, b, g, w, Sig_,(a+w)/(1-a*g^2-b), b+a*g^2, price];%reshape(price', [1,Nstrikes*Nmaturities])];
end
%% Version with weekly szenario generation
scenario_data   = zeros(Nsim*N, 7+Nstrikes*Nmaturities);
count           = 1;
fails = -Nsim;
fprintf('%s','Generatiting Prices. Progress: 0%')
for num_week = 1:N 
    params  = reshape(compare(num_week,:,:),2,4);
    min_    = min(params);
    max_    = max(params);
    for i = 1:Nsim
        price = -ones(1,Nmaturities*Nstrikes);
        while any(any(price < 0)) %|| any(any(price > 1.5*(S-min(K))))
            a = 1;
            b = 1;
            g = 1;
            % Optimize random draw to fit distribution better!!!
            while (b+a*g^2 >= 1)%||(b+a*g^2 <= 0.75)
                a = min_(2)+(max_(2)-min_(2)).*rand(1,1);
                b = min_(3)+(max_(3)-min_(3)).*rand(1,1);
                g = min_(4)+(max_(4)-min_(4)).*rand(1,1);
                fails = fails +1;
            end
            w       = min_(1)+(max_(1)-min_(1)).*rand(1,1);
            %Sig_    = (w+a)/(1-a*g^2-b);
            Sig_    = sig2_0(num_week)*(0.9+0.2*rand(1,1));
            price   = price_Q_clear([w,a,b,g],data_vec,r/252,Sig_);
        end
        if ismember(count,floor(Nsim*N*[1/(5*log10(Nsim*N*100)):1/(5*log10(Nsim*N*100)):1]))
            fprintf('%0.5g',round(count/(N*Nsim)*100,1)),fprintf('%s',"%"),fprintf('\n')
        end
        scenario_data(count,:) = [a, b, g, w, Sig_,(a+w)/(1-a*g^2-b), b+a*g^2, price];%reshape(price', [1,Nstrikes*Nmaturities])];  
        count = count+1;
    end
end
%%
data_price = scenario_data;
save(strcat('data_price_week_',num2str(num_week),'_',num2str(length(data_price)),'_',num2str(r),'_',num2str(min(K)),'_',num2str(max(K)),'_',num2str(min(Maturity)),'_',num2str(max(Maturity)),'.mat'),'data_price')
% Summary
    fprintf('\n')
    disp(strcat("rate/number of pos. price-data: ",num2str((sum(sum((scenario_data(:,8:end)>=0))))/(length(scenario_data)*Nmaturities*Nstrikes)*100),"% , ",num2str((sum(sum((scenario_data(:,8:end)>=0)))))));
    disp(['max price: ',    num2str(max(max(scenario_data(:,8:end))))])
    disp(['min price: ',    num2str(min(min(scenario_data(:,8:end))))])
    disp(['mean price: ',   num2str(mean(mean(scenario_data(:,8:end))))])
    disp(['median price: ', num2str(median(median(scenario_data(:,8:end))))])
    disp(['median alpha: ', num2str(median(scenario_data(:,1)))])
    disp(['median beta: ',  num2str(median(scenario_data(:,2)))])
    disp(['median gamma: ', num2str(median(scenario_data(:,3)))])
    disp(['median omega: ', num2str(median(scenario_data(:,4)))])
    disp(['median sigma: ', num2str(median(scenario_data(:,5)))])
    disp(['median stationary constraint: ', num2str(median(scenario_data(:,7)))])

% Example plot
[X,Y]=meshgrid(K,Maturity);
surf(X',Y',reshape(scenario_data(1,8:end),9,7));hold on;
scatter3(data_vec(:,1),data_vec(:,2),scenario_data(1,8:end));