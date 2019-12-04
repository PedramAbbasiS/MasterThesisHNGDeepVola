%HNG-Optimization under Q via real Optionpath = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
clc; clearvars; close all;
%delete(gcp('nocreate')
%parpool('local',2)
path = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
stock_ind = 'SP500';
year = 2015;
path_ = strcat(path,'/',stock_ind,'/','Calls',num2str(year),'.mat');
load(path_);
bound =[100,100];
omega = 1.8e-9;alpha = 1.5e-6;beta = 0.63;gamma = 250;lambda = 2.4;
sigma0=(alpha+omega)/(1-beta-alpha*gamma.^2);
Init = [omega,alpha,beta,gamma,lambda,sigma0];
% for years = 2010%:2015
%     warning('off','all')
%     surface = surface_generator(years,stock_ind,options.Strikes,options.Maturity,path,bound,0,0,0,0,'linear');
%     [opt_params,opt_params_clean] = hng_params_option(surface,options,Init);
%     warning('on','all')
% end
%NumYears = length(year_files);

% Dates = [731953:7:732310;%2004
%         732317:7:732674;%2005
%         732681:7:733038;%2006
% Dates =         [733045:7:733402;...%2007
%          733409:7:733766;...%2008
%Dates =           [733780:7:734137];%...%2009
%          734144:7:734501;...%2010
%          734508:7:734865;...%2011
%          734872:7:735229;...%2012
%             735236:7:735593];%2013
% Dates =         [733409:7:733766];        
% For Thursdays
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
ms = MultiStart('XTolerance',1e-6,...
    'StartPointsToRun','bounds-ineqs','Display','iter');%,'UseParallel',true);
gs = GlobalSearch(ms,'NumTrialPoints',100,'NumStageOnePoints',20);

r = 0.005/252;
%lb = [1e-12,0,0,-1000,-100,1e-12];%lower parameter bounds
%ub =[1,1,100,2000,100,1]; %upper parameter bounds
lb = [1e-12,0,0,-1000];%lower parameter bounds
ub =[1,1,100,2000]; %upper parameter bounds

data = [OptionsStruct.price;OptionsStruct.maturity;OptionsStruct.strike;OptionsStruct.priceunderlying];
%%
for i = 1%:max(weeksprices)
    data_week = data(:,logical(idx(:,i)))';
    f_min = @(params) sqrt(mean((price_Q(params,data_week,r,Init(6))'-data_week(:,1)).^2));
    problem = createOptimProblem('fmincon','x0',Init(1:4),'objective',f_min,'lb',lb,'ub',ub,'nonlcon',@nonlincon);
    %opt_params_clean(i,:) = run(gs,problem
    opt = optimoptions('fmincon','Display','iter');
    opt_params_clean(i,:) = fmincon(f_min,Init(1:4),[],[],[],[],lb,ub,[],opt);
    %opt_params(i,:)=opt_params_clean(i,:);opt_params(i,3)=opt_params(i,3)+Init(5)+.5;opt_params(i,5) = -0.5;
    %Init = opt_params_clean(i,:);
end

% for i = 1%:max(weeksprices)
%     data_week = data(:,logical(idx(:,i)))';
%     f_min = @(params) sqrt(mean((price(params,data_week,r)'-data_week(:,1)).^2));
%     problem = createOptimProblem('fmincon','x0',Init,'objective',f_min,'lb',lb,'ub',ub,'nonlcon',@nonlincon);
%     opt_params_clean(i,:) = run(gs,problem);
%     opt_params(i,:)=opt_params_clean(i,:);opt_params(i,3)=opt_params(i,3)+Init(5)+.5;opt_params(i,5) = -0.5;
%     Init = opt_params_clean(i,:);
% end


