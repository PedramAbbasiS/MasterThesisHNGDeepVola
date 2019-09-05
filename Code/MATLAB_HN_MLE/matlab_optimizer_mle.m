%% minimizer
% omega = par0(1);
% alpha = par0(2);
% beta = par0(3);
% gamma = par0(4);
% lambda = par0(5);
% h = par0(6);

%% data
data = load('SP500_data.txt');
%idx = (data(:,2)==2012);
%logret = data(idx,4);
r = 0.01/252;
par0 = [2e-6,1e-6,0.67,450,13];
par0_h0 = [2e-6,1e-6,0.67,450,13,1e-5];
lb = [1e-8,0,0,-1000,-100];
lb_h0 = [1e-8,0,0,-1000,-100,1e-8];
ub =[1,1,1-1e-8,2000,100];
ub_h0 =[1,1,1-1e-8,2000,100,1];
A = [];
b = [];
Aeq = [];
beq = [];
Nsim = 1000;
mat_par = zeros(Nsim,6);
mat_par_h0 = zeros(Nsim,6);
mat_ll = zeros(Nsim,1);
mat_ll_h0 = zeros(Nsim,1);
% first date 14 jan 2010 end date 30 dec 2015
win_len = 1750;
for i=1:Nsim
    logret = data(end-Nsim-win_len+1+i:end-Nsim+i,4);
    f_min = @(par) ll_hng_n(par,logret,r);
    f_min_h0 = @(par) ll_hng_n_h0(par,logret,r);
    if i~=1
        par0 = params;
        par0_h0 = params_h0;
    end
    % optimzer without h0 opt
    [params,value] = fmincon(f_min,par0,A,b,Aeq,beq,lb,ub,@nonlincon);
    % optimzier with h0 opt
    [params_h0,value_h0] = fmincon(f_min_h0,par0_h0,A,b,Aeq,beq,lb_h0,ub_h0,@nonlincon);
    sig0 = (params(1)+params(2))/(1-params(4)^2*params(2)-params(3));
    mat_par(i,:) = [params,sig0];
    mat_par_h0(i,:) = params_h0;
    mat_ll(i) = value;
    mat_ll_h0(i) = value_h0;
    if ismember(i,floor(Nsim*[0.025:0.025:1]))
        disp(strcat(num2str(i/Nsim*100),"%"))
    end
end
%%
quant_70_h0 = [quantile(mat_par_h0,0.15,1);quantile(mat_par_h0,0.85,1)];
quant_70 = [quantile(mat_par,0.15,1);quantile(mat_par,0.85,1)];
quant_80_h0 = [quantile(mat_par_h0,0.1,1);quantile(mat_par_h0,0.9,1)];
quant_80 = [quantile(mat_par,0.1,1);quantile(mat_par,0.9,1)];
quant_90_h0 = [quantile(mat_par_h0,0.05,1);quantile(mat_par_h0,0.95,1)];
quant_90 = [quantile(mat_par,0.05,1);quantile(mat_par,0.95,1)];
quant_95_h0 = [quantile(mat_par_h0,0.025,1);quantile(mat_par_h0,0.975,1)];
quant_95 = [quantile(mat_par,0.025,1);quantile(mat_par,0.975,1)];
quant_99_h0 = [quantile(mat_par_h0,0.005,1);quantile(mat_par_h0,0.995,1)];
quant_99 = [quantile(mat_par,0.005,1);quantile(mat_par,0.995,1)];
fun_CI = @(per) [quantile(mat_par,(1-per)/2,1);quantile(mat_par,(1+per)/2,1)];
fun_CI_h0 = @(per) [quantile(mat_par_h0,(1-per)/2,1);quantile(mat_par_h0,(1+per)/2,1)];
%%
% dlmwrite('params_mat_h0.txt',mat_par_h0)
% dlmwrite('params_mat.txt',mat_par)
% save('params_mat_h0','mat_par_h0')
% save('params_mat','mat_par')
% save('quant_95_h0','quant_95_h0')
% save('quant_80_h0','quant_80_h0')
% save('quant_99_h0','quant_99_h0')
% save('quant_70_h0','quant_70_h0')
% save('quant_90_h0','quant_90_h0')
% save('quant_70','quant_70')
% save('quant_80','quant_80')
% save('quant_90','quant_90')
% save('quant_95','quant_95')
% save('quant_99','quant_99')
% save('alltogher_matlab','quant_70','quant_80','quant_90','quant_95','quant_99',...
%     'quant_70_h0','quant_80_h0','quant_90_h0','quant_95_h0','quant_99_h0','mat_par','mat_par_h0')