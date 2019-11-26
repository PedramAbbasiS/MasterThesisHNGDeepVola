function OptionPrice= HestonNandi_Q_oneintegral(S_0,X,Sig_,T,r,w,a,b,g_)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function calculates the price of Call option based on the riskfree
% transformed GARCH  option pricing formula of Heston and Nandi(2000). 
% Input:
% Stock and Option specs: 
%   S_0     current price of the underlying asset
%   K       strike price 
%   Sig_    starting value of variance
%   T       time to maturity in days
%   r       daily risk free interest rate
% Model specs:
%   w       omega
%   a       alpha
%   b       beta
%   g_      riskfree gamma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Henrik Brautmeier
% Date:   Nov 2019
% based on Code of Ali Boloorforoosh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    OptionPrice = .5*(S_0-X*exp(-r*T))+1/pi*exp(-r*T)*integral(@Integrand,eps,1000);
    
    function f=Integrand(phi)
        f = 1./phi.*real(X.^(-1i*phi)*(-1i).*(charac_fun(1i*phi+1)-X.*charac_fun(1i*phi)));
    end

    % function that returns the value for the characteristic function
    function f=charac_fun(phi)
        phi=phi';                    % the input has to be a row vector
        
        % GARCH parameters
        lam_=-.5;                    % risk neutral version of lambda
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





