function [opt_params,opt_params_clean] = hng_params_option(Prices,options,Init)
Maturity = options.Maturity; % vector Maturity in days
K = options.Strikes;
S = options.AssetPrice; 
r = options.rate ; %daily risk free rate
lb = [1e-12,0,0,-1000,-100,1e-12];%lower parameter bounds
ub =[1,1,100,2000,100,1]; %upper parameter bounds
%w = Init(1);a = Init(2);b = Init(3);g = Init(4);lam = Init(5); sig0 = Init(6);
%HNG parameter

m = length(Maturity);
n = length(K);   
f_min = @(params) 1/(m*n)*sum(sum(price(params)-Prices).^2);
ms = MultiStart('XTolerance',1e-9,...
    'StartPointsToRun','bounds-ineqs','Display','iter','UseParallel',true);
gs = GlobalSearch(ms);
problem = createOptimProblem('fmincon','x0',Init,...            
    'objective',f_min,'lb',lb,'ub',ub,'nonlcon',@nonlincon);
[opt_params_clean,~] = run(gs,problem);
opt_params=opt_params_clean;opt_params(3)=opt_params(3)+Init(5)+.5;opt_params(5) = -0.5;
    function p=price(params)
        for t = 1:m
            for k = n:-1:1
                p(t,k)= HestonNandi(S,K(k),params(6),Maturity(t),r,params(1),params(2),params(3),params(4),params(5));
            end
        end
        p = p';
    end
end