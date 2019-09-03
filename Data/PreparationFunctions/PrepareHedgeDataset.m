clear;

load('../Datasets/SP500/Calls2009.mat');
load('../Datasets/SP500/SP500_date_prices_returns.mat');

%%%%%%%Using OptionMetrics%%%%%%%%%% 
% Dates_IS = 730854:7:731211;%2001
% Dates_Thursday = 730855:7:731212;%2001
% Dates_IS = 731218:7:731575;%2002
% Dates_Thursday = 731219:7:731576;%2002
% Dates_IS = 730854:7:731575;%2001-2002
% Dates_Thursday = 730855:7:731576;%2001-2002
% Dates_IS = 731953:7:732310;%2004
% Dates_Thursday = 731954:7:732311;%2004
% Dates_IS = 732317:7:732674;%2005
% Dates_Thursday = 732318:7:732675;%2005
% Dates_IS = 732681:7:733038;%2006
% Dates_Thursday = 732682:7:733039;%2006
% Dates_IS = 733045:7:733402;%2007
% Dates_Thursday = 733046:7:733403;%2007
% Dates_IS = 733409:7:733766;%2008
% Dates_Thursday = 733410:7:733767;%2008 
Dates_IS = 733780:7:734137;%2009
Dates_Thursday = 733781:7:734138;%2009 
% Dates_IS = 734144:7:734501;%2010
% Dates_Thursday = 734145:7:734502;%2010 
% Dates_IS = 734508:7:734865;%2011
% Dates_Thursday = 734509:7:734866;%2011 
% Dates_IS = 734872:7:735229;%2012
% Dates_Thursday = 734873:7:735230;%2012 
% Dates_IS = 735236:7:735593;%2013
% Dates_Thursday = 735237:7:735594;%2013
Type = 'call'; 
TimeToMaturityInterval = [20 250];
MoneynessInterval = [.9 1.1];
MinimumVolume = 1000;
MinimumOpenInterest = 1000;
IfCleanNans = 1;
% r_options = .03388 / 252;%2001
% r_options = .016 / 252;%2002
% r_options = .011 / 252;%2004
% r_options = .03 / 252;%2005
% r_options = .047 / 252;%2006
% r_options = .043 / 252;%2007
% r_options = .016 / 252;%2008
r_options = .0016 / 252;%2009
% r_options = .0011 / 252;%2010
% r_options = .005 / 252;%2011
% r_options = .006 / 252;%2012
% r_options = .0007 / 252;%2013
[OptionData, OptFeatures, Dates, LongestMaturity] = SelectOptions(Dates_IS, Type, TimeToMaturityInterval, MoneynessInterval, MinimumVolume, MinimumOpenInterest,IfCleanNans, ...
    TheDateofthisPriceInSerialNumber, CCallPPut, TradingDaysToMaturity, Moneyness, Volume,OpenInterestfortheOption, StrikePriceoftheOptionTimes1000, MeanOptionPrice, TheSP500PriceThisDate, ...
    TheSP500ReturnThisDate, VegaKappaoftheOption, ImpliedVolatilityoftheOption);
clear TheDateofthisPriceInSerialNumber CCallPPut TradingDaysToMaturity Moneyness Volume OpenInterestfortheOption StrikePriceoftheOptionTimes1000 MeanOptionPrice TheSP500PriceThisDate TheSP500ReturnThisDate VegaKappaoftheOption ExpirationDateoftheOption ExpirationDateoftheOptionInSerialNumber ImpliedVolatilityoftheOption TheDateofthisPrice

% add interest rate for pricing
NumOptions = length(OptionData);
NumDates = length(Dates);
for i = 1:NumOptions
    OptionData(i).r_options = r_options;
end

% add hedging information
HedgingFrequency = 5; % We hedge every HedgingFrequency days
for i = 1:NumOptions
    ind = find(SP500_date_prices_returns(1,:)==OptionData(i).date);
    OptionData(i).TerminalDate = SP500_date_prices_returns(1,ind + OptionData(i).maturity);
    OptionData(i).TerminalPrice = SP500_date_prices_returns(2,ind + OptionData(i).maturity);
    HedgingDates = OptionData(i).date;
    HedgingPrices = OptionData(i).priceunderlying;
    HedgingReturns = OptionData(i).returnunderlying;
    HedgingTtoMat = OptionData(i).maturity;
    DateLastHedgePossible = SP500_date_prices_returns(1,ind + OptionData(i).maturity - HedgingFrequency);
    while HedgingDates(end) < DateLastHedgePossible
        ind = ind + HedgingFrequency;
        HedgingDates = [HedgingDates, SP500_date_prices_returns(1,ind)];
        HedgingPrices = [HedgingPrices, SP500_date_prices_returns(2,ind)];
        HedgingReturns = [HedgingReturns, SP500_date_prices_returns(3,ind)];
        HedgingTtoMat = [HedgingTtoMat HedgingTtoMat(end) - HedgingFrequency];
    end
    HedgingPrices = [HedgingPrices, OptionData(i).TerminalPrice];
    OptionData(i).HedgingDates = HedgingDates;
    OptionData(i).HedgingTtoMat = HedgingTtoMat; 
    OptionData(i).HedgingPrices = HedgingPrices;  
    OptionData(i).HedgingReturns = HedgingReturns;
end
 
% add volatility information
% Estimation period is beginnning 2000 - end 2011, that is from 730854 to 731211
ind = find(and((SP500_date_prices_returns(1,:) >= 730486),(SP500_date_prices_returns(1,:) <= 734865)));
EstimationSample = SP500_date_prices_returns(3,ind);
% NGARCH Gaussian volatility
initParamModelStruct = struct('Drift', 'Constant', 'Variance', 'NGARCH', 'r', 0.0001, 'alpha0', 1e-06,...
    'alpha1', 0.041, 'beta1', 0.917, 'gamma', .863);
initParamDistrStructGauss = struct('DistributionName', 'GAUSSIAN','mu',0,'sigma2',1);
[paramModelStructGarch, paramDistrStructGauss, ~, ~, ~] = GARCHfit(EstimationSample, ...
    initParamModelStruct, initParamDistrStructGauss, []);
[~, NGARCH_Gaussian_sigmas, NGARCH_Gaussian_epsilons, ~] = GARCHlikelihood(SP500_date_prices_returns(3,:), ...
    paramModelStructGarch, paramDistrStructGauss, [], []);
% ARSV symmetric volatility

initParamModelStructSym = struct('r', 0, 'gamma', -0.821, 'phi', 0.9, 'sigma_w', .5);

paramModelStruct_ARSV_sym_Kalman = ARSVfit(EstimationSample, initParamModelStructSym, [], 'Kalman');

[~, ~, ~, ARSV_sym_Kalman_sigma, ~] = ARSVKalmanLikelihood(SP500_date_prices_returns(3,:), paramModelStruct_ARSV_sym_Kalman, []);

paramModelStruct_ARSV_sym_hlik = ARSVfit(EstimationSample, initParamModelStructSym, [], 'h-likelihood');

[~, ~, ~, ~, ~, ARSV_sym_hlik_sigma, ~] = ARSVhlikelihood(SP500_date_prices_returns(3,:), paramModelStruct_ARSV_sym_hlik, [], []);




% ARSV skewed volatility

initParamModelStructSkew = struct('r', 0, 'gamma', -0.821, 'phi', 0.9, 'sigma_w', .5, 'rho', -.6);

paramModelStruct_ARSV_skew_Kalman = ARSVfit(EstimationSample, initParamModelStructSkew, [], 'Kalman');

[~, ~, ~, ARSV_skew_Kalman_sigma, ~] = ARSVKalmanLikelihood(SP500_date_prices_returns(3,:), paramModelStruct_ARSV_skew_Kalman, []);

paramModelStruct_ARSV_skew_hlik = ARSVfit(EstimationSample, initParamModelStructSkew, [], 'h-likelihood');

[~, ~, ~, ~, ARSV_skew_hlik_sigma_forecasted, ARSV_skew_hlik_sigma, ARSV_skew_hlik_sigma_smoothed] = ARSVhlikelihood(SP500_date_prices_returns(3,:), paramModelStruct_ARSV_skew_hlik, [], []);



for i = 1:NumOptions
    numHedges = length(OptionData(i).HedgingDates);
    for j=1:numHedges
        ind = find(SP500_date_prices_returns(1,:)==OptionData(i).HedgingDates(j));
        OptionData(i).NGARCH_Gaussian_sigma(j) = NGARCH_Gaussian_sigmas(ind);
        OptionData(i).NGARCH_Gaussian_epsilon(j) = NGARCH_Gaussian_epsilons(ind);
        OptionData(i).ARSV_sym_Kalman_sigma(j) = ARSV_sym_Kalman_sigma(ind);
        OptionData(i).ARSV_sym_hlik_sigma(j) = ARSV_sym_hlik_sigma(ind);
        OptionData(i).ARSV_skew_Kalman_sigma(j) = ARSV_skew_Kalman_sigma(ind);
        OptionData(i).ARSV_skew_hlik_sigma(j) = ARSV_skew_hlik_sigma(ind);
    end
end

% Order options by Time to Maturity

[~,IX] = sort([OptionData(:).maturity]);
OptionData = OptionData(IX);

