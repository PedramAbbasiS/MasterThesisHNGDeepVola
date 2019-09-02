function [OptionsStruct, OptFeatures, DatesClean, LongestMaturity] = SelectOptions(Dates, Type, ...
    TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans,...
    TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume, ...
    OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
    TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption)
% INPUTS
% Dates is provided in the form of a row vector containing dates in serial date form
% Type can be 'call', 'put', or 'all'
% Moneyness is defined as S/K
% IfCleanNans = 1 the function eliminates the options withNaNs in the Vegas

% OUTPUTS
% OptFeatures is an array of structures labeled by the different dates with the fields:
% -Maturities is a vector with the distinct maturities for that date orderer in increasing order as required by the prices function
% -Strikes is a vector with the distinct strikes for that date
% -Indices: indices in OptionsStruct that correspond to that date
% LongestMaturity is the longest maturity among all the options in all the dates

NumDates = length(Dates);
NumOptions = length(TheDateofthisPriceInSerialNumber);
Selected = zeros(NumOptions, 1);
% Select dates
for i = 1:NumDates
    temp_Selected = (TheDateofthisPriceInSerialNumber == Dates(i));
    Selected = (Selected | temp_Selected);
end
% Select type
if strcmp(Type, 'call')
    Selected = (Selected & strcmp(CCallPPut, 'C'));
elseif strcmp(Type, 'put')
    Selected = (Selected & strcmp(CCallPPut, 'P')); 
end
% Select TimeToMaturityInterval
if ~isempty(TimeToMaturityInterval)
    Selected = (Selected & (TradingDaysToMaturity>=TimeToMaturityInterval(1)));
    Selected = (Selected & (TradingDaysToMaturity<=TimeToMaturityInterval(2)));
end
% Select MoneynessInterval
if ~isempty(MoneynessInterval)
    Selected = (Selected & (Moneyness>=MoneynessInterval(1)));
    Selected = (Selected & (Moneyness<=MoneynessInterval(2)));
end
% Select MinimumVolume
if ~isempty(MinimumVolume)
    Selected = (Selected & (Volume>=MinimumVolume));
end
% Select MinimumVolume
if ~isempty(MinimumOpenInterest)
    Selected = (Selected & (OpenInterestfortheOption>=MinimumOpenInterest));
end
ind = find(Selected);
% Clean NaNs in Vegas
if IfCleanNans
    Selected = (Selected & ~isnan(VegaKappaoftheOption));
end
NumOptions = length(ind);
if NumOptions > 0
    OptionsStruct(NumOptions) = struct('date',0,'maturity',0,'strike',0,'price',0,...
        'priceunderlying',0,'returnunderlying',0,'moneyness',0, 'vega', 0, 'implied_volatility', 0, 'type', Type);
    DatesVector = zeros(NumOptions, 1);
    for i = 1:NumOptions
        OptionsStruct(i).date = TheDateofthisPriceInSerialNumber(ind(i));
        OptionsStruct(i).maturity = TradingDaysToMaturity(ind(i));
        OptionsStruct(i).strike = StrikePriceoftheOptionTimes1000(ind(i)) / 1000;
        OptionsStruct(i).price = MeanOptionPrice(ind(i));
        OptionsStruct(i).priceunderlying = TheSP500PriceThisDate(ind(i));
        OptionsStruct(i).returnunderlying = TheSP500ReturnThisDate(ind(i));
        OptionsStruct(i).vega = VegaKappaoftheOption(ind(i));
        if strcmp(CCallPPut(ind(i)), 'P')
            OptionsStruct(i).type = 'put';
        elseif strcmp(CCallPPut(ind(i)), 'C')
            OptionsStruct(i).type = 'call';
        else
            OptionsStruct(i).type = [];
        end
        OptionsStruct(i).moneyness = Moneyness(ind(i));
        OptionsStruct(i).implied_volatility = ImpliedVolatilityoftheOption(ind(i));
        DatesVector(i) = TheDateofthisPriceInSerialNumber(ind(i));
    end
    
    
    LongestMaturity = 0;
    k = 1;
    for i = 1:NumDates
        ind = find(DatesVector == Dates(i));
        temp_num = length(ind);
        if temp_num > 0
            Maturities = OptionsStruct(ind(1)).maturity;
            Strikes = OptionsStruct(ind(1)).strike;
            for j = 2:temp_num
                maturity = OptionsStruct(ind(j)).maturity;
                if sum(find(maturity == Maturities)) == 0
                    Maturities = [Maturities maturity];
                end
                strike = OptionsStruct(ind(j)).strike;
                if sum(find(strike == Strikes)) == 0
                    Strikes = [Strikes strike];
                end
            end
            [Maturities, ~] = sort(Maturities);
            LongestMaturity = max([LongestMaturity Maturities]);
            OptFeatures(k).Maturities = Maturities;
            OptFeatures(k).Strikes = Strikes;
            OptFeatures(k).Indices = ind;
            DatesClean(k) = Dates(i);
            k = k + 1;
        end
    end
else
    OptionsStruct = [];
    OptFeatures = [];
    DatesClean = [];
    LongestMaturity = [];  
end


% Maturities = OptionsStruct(1).maturity;
% for i = 2:NumOptions
%     maturity = OptionsStruct(i).maturity;
%     Maturities = [Maturities maturity];
% end
% [Maturities,IX] = sort(Maturities);
% OptionsStruct = OptionsStruct(IX);
% Maturities = OptionsStruct(1).maturity;
% Strikes = OptionsStruct(1).strike;
% for i = 2:NumOptions
%     maturity = OptionsStruct(i).maturity;
%     if sum(find(maturity == Maturities)) == 0
%         Maturities = [Maturities maturity];
%     end
%     strike = OptionsStruct(i).strike;
%     if sum(find(strike == Strikes)) == 0
%         Strikes = [Strikes strike];
%     end
% end
