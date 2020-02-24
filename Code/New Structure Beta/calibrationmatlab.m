% Optimizer: ImpVola to HNG Parameters
strikes = 0.9:0.025:1.1;
maturities = 30:30:210;
S = 1;
r = 0.005;
data_vec = [combvec(strikes,maturities);S*ones(1,Nmaturities*Nstrikes)]';
Nstrikes = length(strikes);
Ndata = 10;
Ntrain = 3000;
Nparameters = 5;
prediction_data = rand(Ndata,Nmaturities,Nstrikes);
trainings_data = rand(Ntrain,Nmaturities,Nstrikes);
prediction_data_trafo = reshape(prediction_data,Ndata,Nmaturities*Nstrikes);
xdata = [(1e-9),1,1000,1e-6,1].*rand(Ndata,Nparameters);
xtrain = [(1e-9),1,1000,1e-6,1].*rand(Ntrain,Nparameters);
%% Finding starting values:
error = zeros(Ndata,Ntrain);
init_params = zeros(Ndata,Nparameters);
init_error = zeros(Ndata,1);
for i = 1:Ndata
    for j=1:Ntrain
        error(i,j) = sum(sum((prediction_data(i,:,:)-trainings_data(j,:,:)).^2));
        if j==1
            init_params(i,:) = xtrain(j,:);
            init_error(i) = error(i,j);
        elseif error(i,j)<init_error(i)
            init_params(i,:) = xtrain(j,:);
            init_error(i) = error(i,j);
        end  
    end
end
%% optimization
lb = [0,0,-1000,1e-12,1e-12];
ub = [1,1,1000,1000,2];
opti_params = zeros(Ndata,Nparameters);
%gs = GlobalSearch('XTolerance',1e-9,'StartPointsToRun','bounds-ineqs','Display','final');
opt = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxIterations',20,'MaxFunctionEvaluations',150);    
for i = 1:Ndata
    x0 = init_params(i,:);
    f_min =@(params) fun2opti(params,prediction_data_trafo(i,:),r,data_vec);
    opti_params(i,:) = fmincon(f_min,x0,[],[],[],[],lb,ub,@nonlincon_nn,opt);
    %problem = createOptimProblem('fmincon','x0',x0,...
    %            'objective',f_min,'lb',lb,'ub',ub,'nonlcon',@nonlincon_nn);
    %[xmin,fmin] = run(gs,problem);    
    
end