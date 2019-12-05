%HNG-Optimization under Q 
%Options: path = '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
clc; clearvars; close all;
%delete(gcp('nocreate')
parpool()
path                =  '/Users/User/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
%path                =  '/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';
stock_ind           =  'SP500';
year                =  2015;
path_               =  strcat(path,'/',stock_ind,'/','Calls',num2str(year),'.mat');
load('weekly_2015_mle.mat')
load(path_);
bound               =  [100,100];
Init                =  params_Q_mle_weekly;
formatIn            =  'dd-mmm-yyyy';
DateString_start    =  '01-January-2015';
DateString_end      =  '31-December-2015';
date_start          =  datenum(DateString_start,formatIn);
date_end            =  datenum(DateString_end,formatIn);
Dates               =  date_start:1:date_end;
%Dates = Dates + 1;

Type                = 'call';
MinimumVolume       = 100;
MinimumOpenInterest = 100;
IfCleanNans         = 1;
TimeToMaturityInterval = [30,250];
MoneynessInterval   = [0.9,1.1];
[OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
    TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
    TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
    OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
    TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);
weeksprices         = week(datetime([OptionsStruct.date],'ConvertFrom','datenum'));
idx                 = zeros(length(weeksprices),max(weeksprices));
for i=1:max(weeksprices)
    idx(:,i) = (weeksprices==i);
end
data = [OptionsStruct.price;OptionsStruct.maturity;OptionsStruct.strike;OptionsStruct.priceunderlying];
save('generaldata2015.mat','data','DatesClean','OptionsStruct','OptFeatures','idx')
%% Optiimization

% Initialization     
r                =   0.005/252;
sc_fac           =   magnitude(Init);
Init_scale_mat   =   Init./sc_fac;
lb_mat           =   [1e-12,0,0,-500]./sc_fac;
ub_mat           =   [1,1,10,1000]./sc_fac;
opt_params_raw   =   zeros(max(weeksprices),4);
opt_params_clean =   zeros(max(weeksprices),4);
values           =   cell(1,max(weeksprices));
Init_scale       =   Init_scale_mat(1,:);
scaler           =   sc_fac(1,:);  
% weekly optimization
for i = 1:max(weeksprices)
    data_week = data(:,logical(idx(:,i)))';
    if isempty(data_week)
        continue
    end
    lb = lb_mat(i,:);%lower parameter bounds, scaled
    ub = ub_mat(i,:); %upper parameter bounds, scaled

    % Algorithm
    % RMSE
    %f_min = @(params) sqrt(mean((price_Q(params.*scaler,data_week,r,sig2_0(i))'-data_week(:,1)).^2));
    % MRAE/MAPE
    %%  UPDATE PRICING
    f_min_raw = @(params,scaler) mean(abs(price_Q(params.*scaler,data_week,r,sig2_0(i))'-data_week(:,1))./data_week(:,1));
    % SQP
    opt = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxIterations',20,'MaxFunctionEvaluations',150);
    % Interior Point
    %opt = optimoptions('fmincon','Display','iter','Algorithm','interior-point','MaxIterations',50,'MaxFunctionEvaluations',300,'FunctionTolerance',1e-4);
    
    % Starting value check / semi globalization
    if i~=1
        x1      = Init_scale_mat(i,:);
        scaler  = sc_fac(i,:); 
        f1      = f_min_raw(x1,scaler);
        x2      = opt_params_raw(i-1,:);
        scaler  = scale_tmp;
        f2      = f_min_raw(x2,scaler);
        if f1<f2
            Init_scale = x1;
            scaler     = sc_fac(i,:);
    
        else 
            Init_scale = x2;
            scaler = scale_tmp;
        end
            
    end
    f_min = @(params) f_min_raw(params,scaler); 
    
    
    % updates
    nonlincon_fun = @(params) nonlincon_scale_v2(params,scaler);
    opt_params_raw(i,:) = fmincon(f_min,Init_scale,[],[],[],[],lb,ub,nonlincon_fun,opt);
    opt_params_clean(i,:) = opt_params_raw(i,:).*scaler;
    struc           =   struct();
    struc.Price     =   data_week(:,1)';
    struc.hngPrice  =   price_Q(opt_params_clean(i,:),data_week,r,sig2_0(i)) ;
    struc.blsPrice  =   blsprice(data_week(:,4), data_week(:,3), r*252, data_week(:,2)/252, hist_vola(i), 0)';
    struc.hngparams =   opt_params_clean(i,:);
    struc.countneg  =   sum(struc.hngPrice<=0);
    struc.matr      =   [struc.Price;struc.hngPrice;struc.blsPrice];
    struc.maxAbsEr  =   max(abs(struc.hngPrice-struc.Price));
    struc.MAPE      =   mean(abs(struc.hngPrice-struc.Price)./struc.Price);
    struc.MaxAPE    =   max(abs(struc.hngPrice-struc.Price)./struc.Price);
    struc.RMSE      =   sqrt(mean((struc.hngPrice-struc.Price).^2));
    struc.RMSEbls   =   sqrt(mean((struc.blsPrice-struc.Price).^2));
    struc.scale     =    scaler;
    scale_tmp       =   scaler;
    values{i}       =   struc;
end 
save('params_Options_2015_MRAEfull.mat','values');