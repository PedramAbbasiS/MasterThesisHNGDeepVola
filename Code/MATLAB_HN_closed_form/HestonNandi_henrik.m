function OptionPrice= HestonNandi_henrik(S_0,X,Sig_,T,r,w,a,b,g_)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function calculates the price of Call option based on the GARCH 
% option pricing formula of Heston and Nandi(2000). The input to the
% function are: current price of the underlying asset, strike price,
% unconditional variance of the underlying asset, time to maturity in days,
% and daily risk free interest rate.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Ali Boloorforoosh
% email:  a_bol@jmsb.concordia.ca
% Date:   Nov. 1,08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%%%% sample inputs %%%%%
    % S_0=100;                    stock price at time t
    % X=100;                      strike prices
    % Sig_=.04/252;               unconditional variances per day
    % T=30;                       option maturity
    % r=.05/365;                  daily risk free rate


OptionPrice=.5*S_0+(exp(-r*T)/pi)*integral(@Integrand1,0,2000)-X*exp(-r*T)*(.5+(1/pi)*integral(@Integrand2,0,2000));

    % function Integrand1 and Integrand2 return the values inside the 
    % first and the second integrals
    
    function f1=Integrand1(phi)
        f1=real((X.^(-1i*phi).*charac_fun(1i*phi+1))./(1i*phi));
    end

    function f2=Integrand2(phi)
        f2=real((X.^(-1i*phi).*charac_fun(1i*phi))./(1i*phi));
    end

    % function that returns the value for the characteristic function
    function f=charac_fun(phi)
        phi=phi';         
        % recursion for calculating A(t,T,Phi)=A_ and B(t,T,Phi)=B_
        A(:,T) = zeros(size(phi));
        B(:,T) = zeros(size(phi));
        for i=1:T-1
            A(:,T-i)=A(:,T-i+1)+phi.*r+B(:,T-i+1).*w-.5*log(1-2*a.*B(:,T-i+1));
            B(:,T-i)=-.5*phi+b.*B(:,T-i+1)+(.5*phi.^2-2*a*g_*phi.*B(:,T-i+1)+a*B(:,T-i+1).*g_^2)./(1-2*a*B(:,T-i+1));
        end
        A_=A(:,1)+phi.*r+B(:,1).*w-.5*log(1-2*a.*B(:,1));     % A(t;T,phi)
        B_=-.5*phi+b.*B(:,1)+(.5*phi.^2-2*a*g_*phi.*B(:,1)+a*B(:,1).*g_^2)./(1-2*a*B(:,1)); % B(t;T,phi)
        f=S_0.^phi.*exp(A_+B_.*Sig_);
        f = f';        
    end
end





