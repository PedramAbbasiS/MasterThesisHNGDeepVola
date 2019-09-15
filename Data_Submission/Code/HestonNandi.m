function OptionPrice= HestonNandi(S_0,X,Sig_,T,r,w,a,b,g,lam)

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

%source https://de.mathworks.com/matlabcentral/fileexchange/27644-heston-nandi-option-price

                %%%%% sample inputs %%%%%
    % S_0=100;                    stock price at time t
    % X=100;                      strike prices
    % Sig_=.04/252;               unconditional variances per day
    % T=30;                       option maturity
    % r=.05/365;                  daily risk free rate


OptionPrice=.5*S_0+(exp(-r*T)/pi)*integral(@Integrand1,eps,100)-X*exp(-r*T)*(.5+(1/pi)*integral(@Integrand2,eps,100));

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
        
        phi=phi';    % the input has to be a row vector
        
        lam_=-.5;                   % risk neutral version of lambda
        g_=g+lam+.5;                % risk neutral version of gamma

        
        % recursion for calculating A(t,T,Phi)=A_ and B(t,T,Phi)=B_
        A(:,T-1)=phi.*r;
        B(:,T-1)=lam_.*phi+.5*phi.^2;

        for i=2:T-1
            A(:,T-i)=A(:,T-i+1)+phi.*r+B(:,T-i+1).*w-.5*log(1-2*a.*B(:,T-i+1));
            B(:,T-i)=phi.*(lam_+g_)-.5*g_^2+b.*B(:,T-i+1)+.5.*(phi-g_).^2./(1-2.*a.*B(:,T-i+1));
        end

        A_=A(:,1)+phi.*r+B(:,1).*w-.5*log(1-2.*a.*B(:,1));                    % A(t;T,phi)
        B_=phi.*(lam_+g_)-.5*g_^2+b.*B(:,1)+.5*(phi-g_).^2./(1-2.*a.*B(:,1)); % B(t;T,phi)

        f=S_0.^phi.*exp(A_+B_.*Sig_);
        f=f'; % the output is a row vector

    end

end





