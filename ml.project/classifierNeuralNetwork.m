% classifierNeuralNetwork
%
% Train a neural network classifier using cross validation within
% the training set to decide when to stop. Uses the NetLab package,
% stored in the Netlab subdirectory
%
% Input:
%  - examples, a matrix of real values with dimension of #examples * #features.
%  - labels, a vector of integers with length of #examples.
%
%  ClassifierParameters cell array, containing:
%  - nHidden - how many hidden neurons
%  - nIter   - how many iterations to train for
%              (if 0, use "leave 1 example out" cross-validation
%              within the training set)
%  - optimiz - optimization algorithm used in training
%
% Output:
%  - a trained neural network in the NetLab format (the rest of the
%    framework knows how to deal with it)
%
% Example:
%  - classifier=classifierNeuralNetwork(examples,labels,{20,50,50})


function [net] = classifierNeuralNetwork( examples, labels, classifierParameters)


[nExamples,nFeatures] = size(examples);
sortedLabels = unique(labels);
nLabels = length(sortedLabels);

nIn     = nFeatures;
nHidden = nLabels;
nOut    = nLabels;
outfunc = 'softmax'; % good setting for classification problems
optimiz = 'scg'; % scaled conjugate gradient
iterParams = [];
nIter   = 0; % if 0 -> use CV within the trainining set to find #iterations

k = length(classifierParameters);
if k > 0
    nHidden = classifierParameters{1};
    if k > 1
        nIter = classifierParameters{2};
        if k > 2
            iterParams = classifierParameters{3};
        end
    end
end


% normalize the data and
examples = normalize(examples);

% convert labels to 1-of-N encoding
labelsN  = oneOfNencoding(labels);

% create network
net = mlp( nIn, nHidden, nOut, outfunc );

% find optimal #iterations using 'leave-1-out' cross-validation
nIter = computeOptimalNiter(examples, labels, net, nHidden, nIter, iterParams);

% train network
options     = zeros(1,18);
options(14) = nIter;

net = netopt( net, options, examples, labelsN, optimiz);

end
