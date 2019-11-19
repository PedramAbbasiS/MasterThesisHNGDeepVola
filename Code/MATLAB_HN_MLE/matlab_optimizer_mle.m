%% minimizer
par0 = [2e-6,1e-6,0.67,450,13];
par0_h0 = [2e-6,1e-6,0.67,450,13,1e-5];
omega = par0(1);
alpha = par0(2);
beta = par0(3);
gamma = par0(4);
lambda = par0(5);
h = (alpha+omega)/(1-beta-alpha*gamma.^2);
par_h0(6)=h;
r = 0.005/252;
lb = [1e-12,0,0,-1000,-100];
lb_h0 = [1e-12,0,0,-1000,-1000,-100];
ub =[1,1,100,2000,100];
ub_h0 =[1,1,100,2000,100,100];
A = [];
b = [];
Aeq = [];
beq = [];
%% data
data = load('SP500_data.txt');


Nsim = 250;
mat_par_gs = zeros(Nsim,6);
win_len = 1250;
for i=1:Nsim
    logret = data(end-Nsim-win_len+1+i:end-Nsim+i,4);
    %f_min = @(par) ll_hng_n(par,logret,r);
    f_min_h0 = @(par) ll_hng_n_h0(par,logret,r);
    if i~=1
        %par0 = params;
        par0_h0 = params_h0;
    end
    % optimzer without h0 opt
    %[params,value] = fmincon(f_min,par0,A,b,Aeq,beq,lb,ub,@nonlincon);
    % optimzier with h0 opt
    [params_h0,value_h0] = fmincon(f_min_h0,par0_h0,A,b,Aeq,beq,lb_h0,ub_h0,@nonlincon);
    %sig0 = (params(1)+params(2))/(1-params(4)^2*params(2)-params(3));
    %mat_par(i,:) = [params,sig0];
    %mat_par_h0(i,:) = params_h0;
    %mat_ll(i) = value;
    %mat_ll_h0(i) = value_h0;
    %if ismember(i,floor(Nsim*[0.025:0.025:1]))
    %    disp(strcat(num2str(i/Nsim*100),"%"))
    %end
    gs = GlobalSearch('XTolerance',1e-9,...
    'StartPointsToRun','bounds-ineqs');
    %ms = MultiStart('XTolerance',5e-3,...
    %'StartPointsToRun','bounds-ineqs');
    problem = createOptimProblem('fmincon','x0',par0_h0,...
    'objective',f_min_h0,'lb',lb_h0,'ub',ub_h0,'nonlcon',@nonlincon);
    x = run(gs,problem);
    %x2 = run(ms,problem,20);
    mat_par_gs(i,:) = x;
    %mat_par_ms(i,:) = x2;
end
%%
figure
subplot(1,5,1),plot(mat_par_gs(:,1));set(gca, 'YScale', 'log'),title('omega')
subplot(1,5,2),plot(mat_par_gs(:,2));title('alpha'),
subplot(1,5,3),plot(mat_par_gs(:,3));title('beta')
subplot(1,5,4),plot(mat_par_gs(:,4));title('gamma')
subplot(1,5,5),plot(mat_par_gs(:,5));title('lambda')
%summmary
summary = [mean(mat_par_gs);median(mat_par_gs);std(mat_par_gs);...
    min(mat_par_gs);max(mat_par_gs);quantile(mat_par_gs,0.9);...
    quantile(mat_par_gs,0.1)];
sum_tab = array2table(summary,'VariableNames',{'omega','alpha','beta','gamma','lambda','h0'},...
    'RowNames',{'Mean','Median','Std','min','max','per90','per10'});
disp(sum_tab)


%%
% quant_70_h0 = [quantile(mat_par_h0,0.15,1);quantile(mat_par_h0,0.85,1)];
% quant_70 = [quantile(mat_par,0.15,1);quantile(mat_par,0.85,1)];
% quant_80_h0 = [quantile(mat_par_h0,0.1,1);quantile(mat_par_h0,0.9,1)];
% quant_80 = [quantile(mat_par,0.1,1);quantile(mat_par,0.9,1)];
% quant_90_h0 = [quantile(mat_par_h0,0.05,1);quantile(mat_par_h0,0.95,1)];
% quant_90 = [quantile(mat_par,0.05,1);quantile(mat_par,0.95,1)];
% quant_95_h0 = [quantile(mat_par_h0,0.025,1);quantile(mat_par_h0,0.975,1)];
% quant_95 = [quantile(mat_par,0.025,1);quantile(mat_par,0.975,1)];
% quant_99_h0 = [quantile(mat_par_h0,0.005,1);quantile(mat_par_h0,0.995,1)];
% quant_99 = [quantile(mat_par,0.005,1);quantile(mat_par,0.995,1)];
% fun_CI = @(per) [quantile(mat_par,(1-per)/2,1);quantile(mat_par,(1+per)/2,1)];
% fun_CI_h0 = @(per) [quantile(mat_par_h0,(1-per)/2,1);quantile(mat_par_h0,(1+per)/2,1)];
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


