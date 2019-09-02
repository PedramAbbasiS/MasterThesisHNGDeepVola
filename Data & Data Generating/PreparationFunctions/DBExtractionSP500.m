clear
load('../Datasets/SP500/SP500_date_prices_returns.mat')
%load('../Datasets/DJIA30/DJIA30_date_prices_returns.mat')

conn = database('sandbox','root','root','Vendor','MySQL',...
                'Server','127.0.01', 'PortNumber', 3307);
            
%savefile = 'Puts2010.mat';
            
query = 'SELECT * FROM omeos WHERE secid =108105 AND date >= DATE("2010-01-01") AND date <= DATE("2010-12-31") AND (cp_flag ="P") AND exercise_style = "E" AND volume > 100 AND open_interest > 100 AND vega != "NaN"';     % LIMIT 20';     
%108105 SP500
%102456 DJIA30
% OR cp_flag ="C"
curs = exec(conn,query);
res = fetch(curs);
data = res.Data;
close(conn);
TheDateofthisPrice = data(:, 2);
TheDateofthisPriceInSerialNumber = datenum(TheDateofthisPrice); 
CCallPPut = data(:, 6);
ExpirationDateoftheOption = data(:, 5);
ExpirationDateoftheOptionInSerialNumber = datenum(ExpirationDateoftheOption);
% Compute time to maturities
NumOptions=length(ExpirationDateoftheOptionInSerialNumber);
StrikePriceoftheOptionTimes1000 = cell2mat(data(:, 8));
for i=1:NumOptions
    today=TheDateofthisPriceInSerialNumber(i);
    expiry=ExpirationDateoftheOptionInSerialNumber(i);
    index_today=(SP500_date_prices_returns(1,:)>=today);
    index_expiry=(SP500_date_prices_returns(1,:)<=expiry);
    ind_today_exact=find(index_today,1,'first');
    
    TradingDaysToMaturity(i)=sum((index_today&index_expiry));
    Moneyness(i)=SP500_date_prices_returns(2,ind_today_exact)/(StrikePriceoftheOptionTimes1000(i)/1000);
    
end
TradingDaysToMaturity=TradingDaysToMaturity';
Moneyness=Moneyness';

Volume = cell2mat(data(:, 10));
OpenInterestfortheOption = cell2mat(data(:, 12)); 

for i=1:NumOptions
    ind=find(SP500_date_prices_returns(1,:)==TheDateofthisPriceInSerialNumber(i));
    TheSP500PriceThisDate(i)=SP500_date_prices_returns(2,ind);
    TheSP500ReturnThisDate(i)=SP500_date_prices_returns(3,ind);
    
    
end
TheSP500ReturnThisDate = TheSP500ReturnThisDate';
TheSP500PriceThisDate = TheSP500PriceThisDate';
HighestClosingBidAcrossAllExchanges = cell2mat(data(:, 9));
LowestClosingAskAcrossAllExchanges = cell2mat(data(:, 11));
MeanOptionPrice = [HighestClosingBidAcrossAllExchanges LowestClosingAskAcrossAllExchanges];
MeanOptionPrice = mean(MeanOptionPrice,2);
VegaKappaoftheOption = cell2mat(data(:, 17));
ImpliedVolatilityoftheOption = cell2mat(data(:, 13));

save(savefile, 'TheDateofthisPrice', 'TheDateofthisPriceInSerialNumber', 'CCallPPut', 'ExpirationDateoftheOption', 'ExpirationDateoftheOptionInSerialNumber',...
    'TradingDaysToMaturity', 'Moneyness', 'Volume', 'OpenInterestfortheOption', 'StrikePriceoftheOptionTimes1000', ...
    'TheSP500ReturnThisDate', 'TheSP500PriceThisDate', 'MeanOptionPrice', 'VegaKappaoftheOption', 'ImpliedVolatilityoftheOption')