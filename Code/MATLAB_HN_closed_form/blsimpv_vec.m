function vola = blsimpv_vec(data_vec,r,price)
S =  data_vec(:,3);
K =  data_vec(:,1);
T =  data_vec(:,2);
pool_ = gcp();
pr(1:size(data_vec,1)) = parallel.FevalFuture;
for j =1:size(data_vec,1)
    pr(j) = parfeval(pool_,@blsimpv,1,S(j),K(j),r,T(j)/252,price(j));
end
vola = zeros(1,size(data_vec,1));
for j =1:size(data_vec,1)
    [completedIdx,value] = fetchNext(pr,1); %shutdown after 1s
    vola(completedIdx) = value;
end
cancel(pr)
end