clear all;
% all puts and calls
year_files = {'/Users/Lukas/Documents/GitHub/SeminarOptions/Data/Datasets/SP500/Calls2014.mat'};

%, '../Datasets/SP500/CallsPuts2008.mat',...
%     '../Datasets/SP500/CallsPuts2009.mat', '../Datasets/SP500/CallsPuts2010.mat',...
%     '../Datasets/SP500/CallsPuts2011.mat', '../Datasets/SP500/CallsPuts2012.mat',...
%     '../Datasets/SP500/CallsPuts2013.mat'};
% only calls
% year_files = {'../Datasets/SP500/Calls2007.mat', '../Datasets/SP500/Calls2008.mat',...
%     '../Datasets/SP500/Calls2009.mat', '../Datasets/SP500/Calls2010.mat',...
%     '../Datasets/SP500/Calls2011.mat', '../Datasets/SP500/Calls2012.mat',...
%     '../Datasets/SP500/Calls2013.mat'};
% only puts
% year_files = {'../Datasets/SP500/Puts2007.mat', '../Datasets/SP500/Puts2008.mat',...
%     '../Datasets/SP500/Puts2009.mat', '../Datasets/SP500/Puts2010.mat',...
%     '../Datasets/SP500/Puts2011.mat', '../Datasets/SP500/Puts2012.mat',...
%     '../Datasets/SP500/Puts2013.mat'};

NumYears = length(year_files);

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
%DateString_start = '01-January-2015';
DateString_start = '01-January-2014';
%DateString_end = '31-December-2015';
DateString_end = '31-December-2014';
date_start = datenum(DateString_start,formatIn);
date_end = datenum(DateString_end,formatIn);
Dates = [date_start:1:date_end];

%Dates = Dates + 1;

Type = ['call'];%'call'; 
MinimumVolume = 100;
MinimumOpenInterest = 100;
IfCleanNans = 1;

%MaturitiesBounds = [20 30 80 180 250];
MaturitiesBounds = [15 60 180 250];
%MoneynessBounds = [.95 .975 1 1.025 1.05 1.1];
%MoneynessBounds = [.9 .95 .975 1 1.025 1.05 1.1];
MoneynessBounds = [.9 .97 1.03 1.1];

NumberOfContracts = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);
AveragePrices = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);
AverageImpliedVolatilities = zeros(length(MaturitiesBounds) - 1, length(MoneynessBounds) - 1);








for k = 1:NumYears
    load(year_files{k});
    for i=1:length(MaturitiesBounds) - 1
        for j=1:length(MoneynessBounds) - 1
            TimeToMaturityInterval = [MaturitiesBounds(i) MaturitiesBounds(i + 1)];
            MoneynessInterval = [MoneynessBounds(j) MoneynessBounds(j + 1)];
            OptionData = SelectOptions(Dates(k, :), Type, TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans, TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume,OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);

            
            
            
            if length(OptionData) >0
                NumberOfContracts(i, j) = NumberOfContracts(i, j) + length(OptionData);
                Prices_temp = 0;
                ImpliedVolatilities_temp = 0;
                for r = 1:length(OptionData)
                    Prices_temp = Prices_temp + OptionData(r).price;
                    ImpliedVolatilities_temp = ImpliedVolatilities_temp + OptionData(r).implied_volatility;
                end
                AveragePrices(i, j) = AveragePrices(i, j) + Prices_temp;
                AverageImpliedVolatilities(i, j) = AverageImpliedVolatilities(i, j) + ImpliedVolatilities_temp;
            end
        end
    end  
end

AveragePrices = AveragePrices ./ NumberOfContracts;
AverageImpliedVolatilities = AverageImpliedVolatilities ./ NumberOfContracts;
save('description_data_pricing_thursdays_2015.mat'); % or wednesdays
%save('description_data_pricing_calls_wednesdays_2009.mat');