function p = price_Q_noparallel(params,data_week,r,sig0)
%parloop efficiency variables
w = params(1);
a = params(2);
b = params(3);
g = params(4);
S =  data_week(:,4);
K =  data_week(:,3);
T =  data_week(:,2);
for j =1:size(data_week,1)
    p(j) = HestonNandi_Q(S(j),K(j),sig0,T(j),r,w,a,b,g);
end