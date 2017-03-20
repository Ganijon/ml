%% Normalize each feature to have mean 0 and standard deviation 1

function [Y] = normalize(X)

[nExamples,nFeatures] = size(X);
meanX = mean(X,1);
stdvX = std(X,0,1);

%Y = X -  repmat(meanX,[nExamples,1]);
%Y = Y ./ repmat(stdvX,[nExamples,1]);

Y = X - meanX(ones(nExamples, 1), :);
Y = Y ./ stdvX(ones(nExamples, 1), :);