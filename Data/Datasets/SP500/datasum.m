function s = datasum(Array,flag)
if nargin < 2 || isempty(flag)
    flag = 0;
end
s.mean= mean(Array);
s.median= median(Array);
s.std= std(Array);
s.min = min(Array);
s.max = max(Array);
s.skew = skewness(Array);
s.kurtosis = kurtosis(Array);
s.quant90 = quantile(Array,0.90);
s.quant10 = quantile(Array,0.10);
if flag
    boxplot(Array);
end