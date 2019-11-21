function surface = surface_generator(year,stock_ind,Strikes,Maturity,path,bound,plot,vecormat,saver,volaprice,method)
% Function smooths a volatility surface of data points to a given grid 
% Input: 
%   year:       int             -   year of data 
%   stock_ind:  str             -   IndexName (SP500,NASDAQ100,DJIA30)
%   Strikes,Maturity: double    -   Vector of Strikes/Maturities
%   path:       str             -   local path of data '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets'
%   bound:      double 2x1      -   bounds for minimal Volume and Open Interest
%   plot:       logical         -   show surface plot
%   vecormat:   logical         -   output as vector or matrix
%   saver:      logical         -   save out as .mat-file
%   volaprice:  logical         -   volatility or prices    
%
% Output:
%   surface data
%   optional: .mat-file,surfaceplot
% (c) Henrik Brautmeier, Lukas Wuertenberger 2019

path_ = strcat(path,'/',stock_ind,'/','Calls',num2str(year),'.mat');
if volaprice
    load(path_,'TradingDaysToMaturity','ImpliedVolatilityoftheOption','Moneyness','Volume','OpenInterestfortheOption');
    data = [TradingDaysToMaturity,1./Moneyness,ImpliedVolatilityoftheOption,Volume,OpenInterestfortheOption];
    specs = 'vola';
else
    load(path_,'TradingDaysToMaturity','MeanOptionPrice','Moneyness','Volume','OpenInterestfortheOption','TheSP500PriceThisDate');
    data = [TradingDaysToMaturity,1./Moneyness,MeanOptionPrice./TheSP500PriceThisDate,Volume,OpenInterestfortheOption];
    specs = 'price';
end   
data = data(data(:,5)>=bound(1) &data(:,4)>=bound(1),1:3);
x = data(:,1);
y = data(:,2);
z = data(:,3);
[X,Y] = meshgrid(Maturity,Strikes);
surface_data = griddata(x,y,z,X,Y,method);

if vecormat
    surface = reshape(surface_data,[1,length(Maturity)*length(Strikes)]);
    if saver
        name = strcat('surface',specs,num2str(year),stock_ind);
        save(name,'surface')
        ends
else 
    surface = surface_data;
    if saver
        name = strcat('surface',specs,num2str(year),stock_ind);
        save(name,'surface')
    end
end
if plot 
    figure('Name','RealData')
    scatter3(x,y,z,'filled')
    title(strcat('Scatterplot: Call',specs,' of ',stock_ind,' in ',num2str(year)))
    figure('Name','Fitted Data')
    surf(X,Y,surface_data)
    title(strcat('Scatterplot: Call',specs,' of ',stock_ind,' in ',num2str(year)))
end