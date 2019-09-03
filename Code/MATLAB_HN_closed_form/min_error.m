function y=min_error(phi)
        a =1.1078e-6;
        b=0.7052;
        g=500.88;
        w=1.9835e-6;
        Maturity = 30;
        S = 1;
        K = 0.7:0.05:1.30;
        K = K*S;
        r = 0;
        y = error1(phi,K,S,Maturity,r,w,a,b,g);
end