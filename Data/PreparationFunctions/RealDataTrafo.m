%function surface = realdatasurface(year,stock_ind,path)
path = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
stock_ind = 'SP500';
year = 2013;
Strikes = 0.8:0.025:1.2;
Maturity = 30:30:240;
bound =[100,100];
surface = surface_generator(year,stock_ind,Strikes,Maturity,path,bound,1,1,0);
