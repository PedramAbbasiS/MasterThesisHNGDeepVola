function [c,ceq] = nonlincon(x)
c = x(2)*x(4)^2+x(3)-1+1e-6;
ceq = [];