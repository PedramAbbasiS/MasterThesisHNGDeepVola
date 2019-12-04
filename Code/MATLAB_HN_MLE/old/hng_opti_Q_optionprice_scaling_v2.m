%HNG-Optimization under Q via real Optionpath = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
clc; clearvars; close all;
%delete(gcp('nocreate')
%parpool('local',2)
%path = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
path = '/Users/Henrik/Documents/GitHub/MasterThesisHNGDeepVola/Data/Datasets';

stock_ind = 'SP500';
year = 2015;
load('weekly_2015_mle.mat')
path_ = strcat(path,'/',stock_ind,'/','Calls',num2str(year),'.mat');
load(path_);
bound =[100,100];
Init = params_Q_mle_weekly;
formatIn = 'dd-mmm-yyyy';
DateString_start = '01-January-2015';
DateString_end = '31-December-2015';
date_start = datenum(DateString_start,formatIn);
date_end = datenum(DateString_end,formatIn);
Dates = date_start:1:date_end;

%Dates = Dates + 1;

Type = 'call';
MinimumVolume = 100;
MinimumOpenInterest = 100;
IfCleanNans = 1;
TimeToMaturityInterval = [30,250];
MoneynessInterval = [0.9,1.1];
[OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
    TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
    TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
    OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
    TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);

weeksprices = week(datetime([OptionsStruct.date],'ConvertFrom','datenum'));
idx = zeros(length(weeksprices),max(weeksprices));
for i=1:max(weeksprices)
    idx(:,i) = (weeksprices==i);
end
%ms = MultiStart('XTolerance',1e-6,...
%    'StartPointsToRun','bounds-ineqs','Display','iter');%,'UseParallel',true);
%gs = GlobalSearch(ms,'NumTrialPoints',100,'NumStageOnePoints',20);

r = 0.005/252;
%lb = [1e-12,0,0,-1000,-100,1e-12];%lower parameter bounds
%ub =[1,1,100,2000,100,1]; %upper parameter bounds

data = [OptionsStruct.price;OptionsStruct.maturity;OptionsStruct.strike;OptionsStruct.priceunderlying];
%%
sc_fac = magnitude(Init);
Init_scale_mat = Init./sc_fac;
lb_mat = [1e-12,0,0,-500]./sc_fac;
ub_mat = [1,1,10,1000]./sc_fac;
opt_params_raw = zeros(max(weeksprices),4);
opt_params_clean = zeros(max(weeksprices),4);
values = cell(1,max(weeksprices));
Init_scale = [];
for i = 1:max(weeksprices)
    data_week = data(:,logical(idx(:,i)))';
    if isempty(data_week)
        continue
    end
    lb = lb_mat(i,:);%lower parameter bounds, scaled
    ub = ub_mat(i,:); %upper parameter bounds, scaled

    %RMSE
    %f_min = @(params) sqrt(mean((price_Q(params.*sc_fac(i,:),data_week,r,sig2_0(i))'-data_week(:,1)).^2));
    %MRAE
    f_min = @(params) mean(abs(price_Q(params.*sc_fac(i,:),data_week,r,sig2_0(i))'-data_week(:,1))./data_week(:,1));
    
    % Starting value check / semi globalization
    if isempty(Init_scale)
        Init_scale = Init_scale_mat(i,:);
    else 
        f1 = f_min(Init_scale);
        f2 = f_min(Init_scale_mat(i,:));
        if f2<f1
             Init_scale = Init_scale_mat(i,:);
        end
    end
    
    %SQP
    opt = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxIterations',20,'MaxFunctionEvaluations',150);
    % Interior Point
    %opt = optimoptions('fmincon','Display','iter','Algorithm','interior-point','MaxIterations',50,'MaxFunctionEvaluations',300,'FunctionTolerance',1e-4);
    
    nonlincon_fun = @(params) nonlincon_scale_v2(params,sc_fac(i,:));
    opt_params_raw(i,:) = fmincon(f_min,Init_scale,[],[],[],[],lb,ub,nonlincon_fun,opt);
    opt_params_clean(i,:) = opt_params_raw(i,:).*sc_fac(i,:);
    struc = struct();
    struc.Price = data_week(:,1)';
    struc.hngPrice =    price_Q(opt_params_clean(i,:),data_week,r,sig2_0(i)) ;
    struc.blsPrice =    blsprice(data_week(:,4), data_week(:,3), r*252, data_week(:,2)/252, hist_vola(i), 0)';
    struc.hngparams =   opt_params_clean(i,:);
    struc.countneg =    sum(struc.hngPrice<=0);
    struc.matr =        [struc.Price;struc.hngPrice;struc.blsPrice];
    struc.maxAbsEr =    max(abs(struc.hngPrice-struc.Price));
    struc.MAPE =        mean(abs(struc.hngPrice-struc.Price)./struc.Price);
    struc.MaxAPE =    max(abs(struc.hngPrice-struc.Price)./struc.Price);
    struc.RMSE =       sqrt(mean((struc.hngPrice-struc.Price).^2));
    struc.RMSEbls =       sqrt(mean((struc.blsPrice-struc.Price).^2));
    values{i} =struc;
    Init_scale = opt_params_raw(i,:);
end
save('params_Options_2015_MRAE.mat','values3');