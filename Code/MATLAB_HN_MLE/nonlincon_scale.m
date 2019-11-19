function [c,ceq] = nonlincon_scale(x)
c = (x(2)*(1e-5)).*((x(4)*1000).^2)+x(3)-1+1e-6;
ceq = [];