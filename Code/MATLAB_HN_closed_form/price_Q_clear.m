function p = price_Q_clear(params,data_vec,r,sig0)
% r is daily rate
%parloop efficiency variables
w = params(1);
a = params(2);
b = params(3);
g = params(4);
S =  data_vec(:,3);
K =  data_vec(:,1);
T =  data_vec(:,2);
pool_ = gcp();
pr(1:size(data_vec,1)) = parallel.FevalFuture;
%p=max(S-K,0)';%maximal intrinsic value
p = exp(-r*T).*K; %upper bound call
p = p';
for j =1:size(data_vec,1)
    pr(j) = parfeval(pool_,@HestonNandi_Q_oneintegral,1,S(j),K(j),sig0,T(j),r,w,a,b,g);
end
for j =1:size(data_vec,1)
    [completedIdx,value] = fetchNext(pr,0.5); %shutdown after 0.5s for integral calc
    p(completedIdx) = value;
end
cancel(pr)
end