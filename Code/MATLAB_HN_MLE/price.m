function p = price(params,data_week,r)
for j =1:size(data_week,1)
    p(j) = HestonNandi(data_week(j,4),data_week(j,3),params(6),data_week(j,2),r,params(1),params(2),params(3),params(4),params(5));
end
end