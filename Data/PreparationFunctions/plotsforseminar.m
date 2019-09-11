%function surface_generator(year,stock_ind,Strikes,Maturity,path,bound,plot,vecormat,saver,volaprice,method)
close all;
path = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
%path ='/Users/Lukas/Documents/GitHub/SeminarOptions/Data/Datasets';
stock_ind = 'SP500';
Strikes = 0.9:0.02:1.1;
Maturity = 30:30:240;
year = 2015;
Nstrikes = length(Strikes);
Nmaturities = length(Maturity);
S0 = 1;
bound =[100,100];
path_ = strcat(path,'/',stock_ind,'/','Calls',num2str(year),'.mat');
load(path_,'TradingDaysToMaturity','MeanOptionPrice','Moneyness','Volume','OpenInterestfortheOption','TheSP500PriceThisDate');
data = [TradingDaysToMaturity,1./Moneyness,MeanOptionPrice./TheSP500PriceThisDate,Volume,OpenInterestfortheOption];
data = data(data(:,5)>=bound(1) &data(:,4)>=bound(1),1:3);
x = data(:,1);
y = data(:,2);
z = data(:,3);
[X,Y] = meshgrid(Maturity,Strikes);
method = 'cubic';
surface_data = griddata(x,y,z,X,Y,method);
%%
figure('Name','RealData')
%subplot(1,2,1)
idx =  y<=1.1 &y>=0.9 &x>=30 &x<=240;
scatter3(x(idx),y(idx),z(idx),'.','red')
hold on
%subplot(1,2,2)
surf(X,Y,surface_data)
xlabel('Maturity in Days'),ylabel('relative Strike'),zlabel('Option Price')
xlim([30,240])
ylim([0.9,1.1])
export_fig('C:/Dokumente/test.png','-zbuffer','-r600')