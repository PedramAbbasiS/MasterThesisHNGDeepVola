function [c,ceq] = nonlincon_nn(x)
c = x(1)*x(3)^2+x(2)-1+1e-6;
ceq = [];