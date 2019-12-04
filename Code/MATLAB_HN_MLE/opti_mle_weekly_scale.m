%matlab_optimizer_mle_v2
clc; close all; clearvars;
%% Initialziation
omega = 1.8e-9;alpha = 1.5e-6;beta = 0.63;gamma = 250;lambda = 2.4;
sigma0=(alpha+omega)/(1-beta-alpha*gamma.^2);
Init = [omega,alpha,beta,gamma,lambda,sigma0];
r = 0.005/252;
lb_h0 = [1e-12,0,0,-1000,-100,1e-12];
ub_h0 =[1,1,100,2000,100,1];
A = [];
b = [];
Aeq = [];
beq = [];
%nonlincon contains nonlinear constraint
%% yearly data
data = load('SP500_data.txt');
dates = [data(:,2),week(datetime(data(:,1),'ConvertFrom','datenum'))];
year=2015;
win_len = 2520; %around 10years
num_week = max(dates(dates(:,1)==year,2));
opt_ll = NaN*ones(num_week,1);
params_mle_weekly=NaN*ones(num_week,6);
hist_vola = NaN*ones(num_week,1);
%% optimization
% TODO SCALE PARAMETERS AS IN CALLOPTI!!!!!
for i=1:num_week
    indicee = find(((dates(:,2)==i).*(dates(:,1)==year))==1,1,'first');
    if isempty(indicee)
        continue
    end
    logret_1y = data(indicee-252:indicee-1,4);
    hist_vola(i) = sqrt(252)*std(logret);
    logret = data(indicee-win_len:indicee-1,4);
    f_min = @(par) ll_hng_n_h0(par,logret,r);
    gs = GlobalSearch('XTolerance',1e-9,'StartPointsToRun','bounds-ineqs','Display','final');
    if i~=1 
        x0=[params;Init];
        fmin_ = zeros(2,1);
        xmin_ = zeros(2,6);
        for j=1:2
            problem = createOptimProblem('fmincon','x0',x0(j,:),...
                'objective',f_min,'lb',lb_h0,'ub',ub_h0,'nonlcon',@nonlincon);
            [xmin_(j,:),fmin_(j)] = run(gs,problem);
        end
        [fmin,idx] = min(fmin_);
        xmin = xmin_(idx,:);
    else
        gs = GlobalSearch('XTolerance',1e-9,...
            'StartPointsToRun','bounds-ineqs','NumTrialPoints',2e3);
        problem = createOptimProblem('fmincon','x0',Init,...
                'objective',f_min,'lb',lb_h0,'ub',ub_h0,'nonlcon',@nonlincon);
        [xmin,fmin] = run(gs,problem);    
    end    
    params =xmin;
    opt_ll(i)= fmin;
    params_mle_weekly(i,:)=xmin;
end
params_Q_mle_weekly = [params_mle_weekly(:,1),params_mle_weekly(:,2),params_mle_weekly(:,3),params_mle_weekly(:,4)+params_mle_weekly(:,5)+0.5];
sig2_0 = params_mle_weekly(:,6);
%save('weekly_2015_mle.mat','sig2_0','hist_vola','params_Q_mle_weekly')