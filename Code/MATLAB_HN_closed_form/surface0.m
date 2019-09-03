clearvars 
Maturity = 5:5:150;
K = 0.7:0.025:1.30;
Nmaturities = length(Maturity);
Nstrikes = length(K);
S = 1;
%r = 0.025/252;
r = 0;
j = 0;
l = 0;
Nsim = 1 ;
scenario_data = zeros(Nsim, 7+Nstrikes*Nmaturities);
sig_mat = zeros(Nsim,1);
sig_mat_rn = zeros(Nsim,1);
fprintf('%s','Generatiing Progress: 0%')
for i = 1:Nsim
    if ismember(i,floor(Nsim*[1/(10*log10(Nsim*100)):1/(10*log10(Nsim*100)):1]))
        fprintf('%0.5g',i/Nsim*100),fprintf('%s',"%")
    end
    price = -ones(Nmaturities,Nstrikes);
    while any(any(price < 0)) || any(any(price > 0.45))
        a = 1;
        b = 1;
        g = 1;
        while (b+a*g^2 > 1)
            a = 1e-8 + (1e-6-1e-8).*rand(1,1);
            b = .5 + (.85-.5).*rand(1,1);
            g = 400 + (550-400).*rand(1,1);
            % 95% quantil mit h(0) optimierung
            %a = 1.08e-8 + (2.35e-6-1.08e-8).*rand(1,1);
            %b = .43 + (.97-.43).*rand(1,1);
            %g = 453 + (477-453).*rand(1,1);
            % 95% quantil ohne h(0) optimierung
            %a = 5.8e-7 + (1.4e-6-5.8e-7).*rand(1,1);
            %b = .43 + (.75-.43).*rand(1,1);
            %g = 441 + (590-441).*rand(1,1);
        end
        w = 9e-7 + (3.45e-4-7.55e-6).*rand(1,1);
        %Sig_ = 1e-7 + (1e-3-1e-7).*rand(1,1);
        % 95% quantil mit h(0) optimierung
        %w = 1.6e-6 + (3.2e-6-1.6e-6).*rand(1,1);
        %Sig_ = 4.5e-5 + (1e-3-4.5e-5).*rand(1,1);
        % 95% quantil ohne h(0) optimierung
        %w = 4.1e-7 + (2.9e-6-4.1e-7).*rand(1,1);
        var = 0.3;
        Sig_ = (w+a)/(1-b-a*g^2)*(1-var+2*var*rand(1,1));
        sig_mat(i) = (w+a)/(1-b-a*g^2);
        sig_mat_rn(i) = Sig_;
        lam = 0.0;
        for t = 1:Nmaturities
            for k = 1:Nstrikes
                price(t,k)= HestonNandi(S,K(k),Sig_,Maturity(t),r,w,a,b,g,lam);
            end
        end
    end
    scenario_data(i,:) = [a, b, g, w, Sig_,(a+w)/(1-a*g^2-b), b+a*g^2, reshape(price', [1,Nstrikes*Nmaturities])];  
end
%%
prices_all = scenario_data(:, 8:end);
iv = zeros(size(scenario_data(:,8:end)));
iv_p = zeros(Nstrikes,Nmaturities);
fprintf('%s','ImpVola Progress: 0%')
for i = 1:Nsim
    if ismember(i,floor(Nsim*[1/(10*log10(Nsim*100)):1/(10*log10(Nsim*100)):1]))
        fprintf('%0.5g',i/Nsim*100),fprintf('%s',"%")
    end
    price_new = reshape(prices_all(i,:), [Nstrikes, Nmaturities]);
    for t = 1:Nmaturities
        for k = 1:Nstrikes
            iv_p(k,t) = blsimpv(S,K(k),r,Maturity(t)/252,price_new(k,t));
        end
    end
iv(i,:) =  reshape(iv_p, [1,Nstrikes*Nmaturities]);
end
figure
surf(iv_p)
zlim([0,1])