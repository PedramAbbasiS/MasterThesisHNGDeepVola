%function surface_generator(year,stock_ind,Strikes,Maturity,path,bound,plot,vecormat,saver,volaprice,method)
close all;
path = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
%path ='/Users/Lukas/Documents/GitHub/SeminarOptions/Data/Datasets';
stock_ind = 'SP500';
Strikes = 0.9:0.02:1.1;
Maturity = 30:30:240;
bound =[100,100];
for year = 2010:2015
    surface = surface_generator(year,stock_ind,Strikes,Maturity,path,bound,1,1,1,0,'linear');
end
log_returns= log(1+SP500_date_prices_returns(3,:));