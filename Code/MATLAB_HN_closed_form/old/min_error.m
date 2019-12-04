%
close all;clc;clearvars;
path = '/Users/User/Documents/GitHub/SeminarOptions/Data/Datasets';
stock_ind = 'SP500';
year = 2015;
Strikes = 0.8:0.025:1.2;
Maturity = 30:60:210;
%Maturity =60;
bound =[100,100];
surface = surface_generator(year,stock_ind,Strikes,Maturity,path,bound,0,0,0,0);
%curve = surface(:,1);
%%
S = 1;
Strikes = S*Strikes;
surface =surface*S;
r = 0.005;
%a0= 1.51e-6;
a0=0.000005;
%b0 = 0.662;
b0 = 0.85;
%g0 = 464.2;
g0 = 150;
sig0= 0.05/252;
%w0 = 4.29e-7;
w0 = sig0*(1-b0-a0*g0^2)-a0;
%sig0 =(w0+a0)/(1-b0-a0*g0^2);
%lb = [1e-6,0.0,1e-7,0.5,400,1e-7];
lb=[1e-7,1e-8,0,0.6,100];
ub=[1e-2,1e-3,1e-3,1,800];
%ub = [1e-3,0.1,1e-5,0.9,520,1e-6];
x0 = [sig0,w0,a0,b0,g0];
prices0 =hn_matrix(x0(1),Strikes,S,Maturity,r,x0(2),x0(3),x0(4),x0(5));
%%
clc;
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
%problem = createOptimProblem('fmincon','objective',...
%    fun,'x0',3,'lb',lb,'ub',ub,,'options',options);
%gs = GlobalSearch;
%[x,f] = run(gs,problem)
args=fmincon(@(x) min_fit(x(1),Strikes,S,Maturity,r,x(2),x(3),x(4),x(5),surface),x0,[],[],[],[],lb,ub,@(x) nonlincon(x),options);
prices_opt =hn_matrix(args(1),Strikes,S,Maturity,r,args(2),args(3),args(4),args(5));
rel_error = (surface-prices_opt)./surface*100;
rel_error2 = (surface-prices0)./surface*100;
iv = zeros(length(Strikes),length(Maturity));imp_iv = zeros(length(Strikes),length(Maturity));iv_init=zeros(length(Strikes),length(Maturity));
for t = 1:length(Maturity)
    for k = 1:length(Strikes)
        imp_iv(k,t) = blsimpv(S,Strikes(k),0.005,Maturity(t)/252,surface(k,t),10,0,1e-8,true);
        if prices0(k,t)>=0
            iv_init(k,t) = blsimpv(S,Strikes(k),0.005,Maturity(t)/252,prices0(k,t),10,0,1e-8,true);
        else 
            iv_init(k,t) = NaN;
        end
        if prices_opt(k,t)>=0
            iv(k,t) = blsimpv(S,Strikes(k),0.005,Maturity(t)/252,prices_opt(k,t),10,0,1e-8,true);
        else 
            iv(k,t) = NaN;
        end
    end
end
%%
[X,Y] =meshgrid(Maturity,Strikes);
figure,surf(X,Y,surface);hold on,surf(X,Y,prices_opt),hold on,surf(X,Y,prices0);legend('Surface','opt','initial')
figure,surf(X,Y,iv);hold on, surf(X,Y,imp_iv);hold on, surf(X,Y,iv_init);
figure,surf(X,Y,rel_error);hold on, surf(X,Y,rel_error2);




function [c,ceq] = nonlincon(x)
c(1) = x(3)*x(5)^2+x(4)-1+1e-6;
%c(2) = -x(3)*x(5)^2+x(4)+0.75;
ceq = [];
end