[data, textdata] = importdata('SP500.csv',',');
Dates = data.textdata(3:end,1);
Dates_SP500 = datenum(Dates,'yyyy-mm-dd');
Prices = data.data(:,5);
Returns_all = price2ret(Prices');
SP500_date_prices_returns(3,:) = [SP500_date_prices_returns(3,:), Returns_all(132:end)];

% 
% 
% ind = zeros(length(Dates), 1);
% for i=1:length(DatesCov)
%     ind = ind + (DatesCov(i) == Dates(:));
%     
% end
% Prices = Prices(logical(ind));
% PFE_price = flipud(Prices);
% 
% 
% Prices_all = [AAPL_price'; ABT_price'; AXP_price'; BA_price'; BAC_price'; BMY_price'; BP_price'; C_price'; CAT_price';...
% CSCO_price'; CVX_price'; DELL_price'; DIS_price'; EKDKQ_price'; EXC_price'; F_price'; GE_price';...
% HD_price'; HON_price'; IBM_price'; INTC_price'; JNJ_price'; JPM_price'; KO_price';LLY_price';MCD_price';...
% MMM_price';MSI_MOT_price';MRK_price';MS_price';MSFT_price';ORCL_price';PG_price';PFE_price';SLB_price';T_price';TWX_price';VZ_price';...
% WFC_price';WMT_price';XOM_price';XRX_price'];
% 
% 
% save('Quotes/quotes_prices_datesCov.mat','Prices_all','-append');
% 
% 
% Returns_all = Returns_all';
% 
% save('Quotes/quotes_prices_datesCov.mat','Returns_all','-append');

