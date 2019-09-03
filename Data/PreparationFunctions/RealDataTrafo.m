%function surface = realdatasurface(year,stock_ind,path)
path = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
stock_ind = 'SP500';
year = 2015;
path_ = strcat(path,'/',stock_ind,'/','Calls',num2str(year),'.mat');
load(path_);
data = [TradingDaysToMaturity,ImpliedVolatilityoftheOption,1./Moneyness,Volume,OpenInterestfortheOption];
data = data(data(:,5)>=100 &data(:,4)>=100,1:3);
%data = data(data(:,1)>=25 & data(:,1)<=365,:);
Strikes = 0.8:0.05:1.2;
Maturity = 30:5:240;
surface = zeros(length(Strikes),length(Maturity));
l = 0; error = 0;
for k=Strikes
    s = 0;
    l = l+1;
    for t=Maturity
        volas = data(data(:,1)>=t-2 &data(:,1)<=t+2 & data(:,3)>=k-0.02 &data(:,3)<=k+0.02  ,2);
        if isempty(volas)
            error = error+1;
        end
        s = s+1;
        surface(l,s) = mean(volas);
    end
end
%%
x = data(:,1);
y = data(:,3);
z = data(:,2);
%disp(error/(l*s))
figure
[X,Y] = meshgrid(Maturity,Strikes);
surf(X,Y,surface)
% hold on
figure


figure
Maturity_better = 30:30:240;
Strikes_better = 0.8:0.05:1.2;
[X1,Y2] = meshgrid(Maturity_better,Strikes_better);
scatter3(x,y,z,'filled'); hold on
%get(gcf, 'Renderer')
surface_data = griddata(x,y,z,X1,Y2);
surf(X1,Y2,surface_data)
get(gcf, 'Renderer')
vector_surface = reshape(surface_data,[1,length(Maturity_better)*length(Strikes_better)]);
save('surface_sp500_new','vector_surface')


