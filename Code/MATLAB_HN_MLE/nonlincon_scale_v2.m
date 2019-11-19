function [c,ceq] = nonlincon_scale_v2(x,scale)
c = (x(2)*scale(2)).*((x(4)*(scale(4))).^2)+x(3)*scale(3)-1+1e-6;
ceq = [];