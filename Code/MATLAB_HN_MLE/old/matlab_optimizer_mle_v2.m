%matlab_optimizer_mle_v2
clc; close all; clearvars;
%% minimizer
omega = 1.8e-9;alpha = 1.5e-6;beta = 0.63;gamma = 250;lambda = 2.4;
sigma0=(alpha+omega)/(1-beta-alpha*gamma.^2);
par0_h0 = [omega,alpha,beta,gamma,lambda,sigma0];
%par0_h0 = params_h0;
r = 0.005/252;
lb_h0 = [1e-12,0,0,-1000,-100,1e-12];
ub_h0 =[1,1,100,2000,100,1];
A = [];
b = [];
Aeq = [];
beq = [];
%nonlincon contains nonlinear constraint
%% data
data = load('SP500_data.txt');
win_len = 2500;
year=2010:2015;
mat_par_gs = zeros(length(year),6);
mat_fmin = zeros(1,length(year));
indicee = zeros(size(year));
for i=1:length(year)
    indicee = find(data(:,2)==year(i),1,'first');
    logret = data(indicee-win_len:indicee-1,4);
    f_min_h0 = @(par) ll_hng_n_h0(par,logret,r);
    gs = GlobalSearch('XTolerance',1e-9,...
    'StartPointsToRun','bounds-ineqs');%,'NumTrialPoints',10e3);
    ms = MultiStart('XTolerance',5e-3,...
    'StartPointsToRun','bounds-ineqs');
    if i~=1 
        x0=[par0_h0;params_h0];
        for j=1:2
            problem = createOptimProblem('fmincon','x0',x0(j,:),...
                'objective',f_min_h0,'lb',lb_h0,'ub',ub_h0,'nonlcon',@nonlincon);
            [xmin_(j,:),fmin_(j)] = run(gs,problem);
            %[xmin,fmin,flag,outpt,allmins] = run(gs,problem);
            %[xmin,fmin,flag,outpt,allmins] = run(ms,problem,1000);
        end
        [fmin,idx] = min(fmin_);
        xmin = xmin_(idx,:);
    else
        gs = GlobalSearch('XTolerance',1e-9,...
            'StartPointsToRun','bounds-ineqs','NumTrialPoints',2e3);
        problem = createOptimProblem('fmincon','x0',par0_h0,...
                'objective',f_min_h0,'lb',lb_h0,'ub',ub_h0,'nonlcon',@nonlincon);
        [xmin,fmin] = run(gs,problem);    
    end    
    params_h0 =xmin;
    mat_fmin(i)= fmin;
    mat_par_gs(i,:)=xmin;
end
%%
% plot(arrayfun(@(x)f_min_h0(x.X),allmins),'k*')
% xlabel('Solution number')
% ylabel('Function value')
% title('Solution Function Values')
%%
% %summmary
% summary = [mean(mat_par_gs);median(mat_par_gs);std(mat_par_gs);...
%     min(mat_par_gs);max(mat_par_gs);quantile(mat_par_gs,0.9);...
%     quantile(mat_par_gs,0.1)];
% sum_tab = array2table(summary,'VariableNames',{'omega','alpha','beta','gamma','lambda','h0'},...
%     'RowNames',{'Mean','Median','Std','min','max','per90','per10'});
% disp(sum_tab)
% 
% 
