%
% Cross validation within training set to determine how many
% iterations to train the network for
%

function [nIter] = computeOptimalNiter(examples, labels, net, nHidden, nIter, iterParams )

sortedLabels = unique(labels);
nLabels = length(sortedLabels);

for l=1:nLabels
    label   = sortedLabels(l);
    
    if nIter == 0
        % leave examples out
        
        places     = find(labels == label);
        nblocks(l) = length(places);
        for p=1:nblocks(l)
            orgSets{l}{p} = [places(p) places(p)];
        end
        
    elseif nIter == -1
        % leave 1 block out - HACK, DON'T USE
        
        %% Creates a data structure, orgSets{#labels}, where each entry is a
        %% cell array of intervals of indices into the array of
        %% examples. Each interval contains the beginning and end of
        %% the group of examples.
        
        % for the original examples
        % find the indices of the first image in each
        % block/presentation for this condition
        places  = find(labels == label);
        lastp   = find(diff(places)>1);
        breakp  = lastp + 1;
        breakp  = [1; breakp]; % block beginnings
        lastp   = [lastp; length(places)];
        nblocks(l) = length(breakp);
        
        % now compute the index intervals in the example array
        % corresponding to each block of examples
        orgSets{l} = cell(nblocks(l),1);
        for p=1:nblocks(l)
            orgSets{l}{p} = [places(breakp(p)) places(lastp(p))];
        end
    else
        fprintf('error: nIter=%d is not supported\n',nIter);pause;return
    end
    
end; % for over labels

%% check that all labels have the same number of trials
if sum(diff(nblocks))
    fprintf('error: # of trials should be the same for all labels\n');
    return;
else
    nTrialsPerLabel = nblocks(1);
end

%% Create a structure that says which examples are to be used in each fold.

if isempty(iterParams)
    nFolds = nTrialsPerLabel;
else
    nFolds = iterParams(1);
end

% Identify the image numbers that are going to be used
% as train and test in each fold
trainImagesPerFold    = cell(nFolds,1);
trainIntervalsPerFold = cell(nFolds,1);
testImagesPerFold     = cell(nFolds,1);
testIntervalsPerFold  = cell(nFolds,1);

for k=1:nFolds
    
    % find the test intervals
    testImagesPerFold{k} = [];
    testIntervalsPerFold{k} = {};
    
    for l=1:nLabels
        ii = orgSets{l}{k}; % image interval
        testImagesPerFold{k} = [testImagesPerFold{k},ii(1):ii(2)];
        testIntervalsPerFold{k}{l} = ii;
    end
    
    % find the train intervals
    trainImagesPerFold{k} = [];
    trainIntervalsPerFold{k} = {};idx = 1;
    
    for ak=1:k-1
        for l=1:nLabels
            ii = orgSets{l}{ak}; % image interval
            trainImagesPerFold{k} = [trainImagesPerFold{k},ii(1):ii(2)];
            trainIntervalsPerFold{k}{idx} = ii; idx = idx + 1;
        end
    end
    
    for ak=k+1:nFolds
        for l=1:nLabels
            ii = orgSets{l}{ak}; % image interval
            trainImagesPerFold{k} = [trainImagesPerFold{k},ii(1):ii(2)];
            trainIntervalsPerFold{k}{idx} = ii; idx = idx + 1;
        end
    end
end; % for over folds

%% Run the cross validation
%% - the examples used for train/test in each fold were defined in 4

% run k-fold cross validation

nIterPerFold = zeros(1,nFolds);
errorPerFold = zeros(1,nFolds);
size(labels)

for k=1:nFolds
    fprintf('Testing over fold %d\n',k);
    % a) select examples for test fold and train folds
    trainExamples   = examples(trainImagesPerFold{k},:);
    trainLabels     = labels(trainImagesPerFold{k},:);
    trainLabels1ofN = oneOfNencoding(trainLabels);
    testExamples    = examples(testImagesPerFold{k},:);
    testLabels      = labels(testImagesPerFold{k},:);
    testLabels1ofN  = oneOfNencoding(testLabels);
    
    % set up training
    nIterPerBurst   = 10;
    nBursts         = 20;
    errorAfterBurst{k} = zeros(1,nBursts);
    
    % train the network in little bursts
    foldNet = net;
    options = zeros(1,18);
    options(1)  = 1;                 % Print out error values
    options(14) = nIterPerBurst;
    method = 'scg';
    for b = 1:nBursts
        % train for <iterationsPerBurst>
        
        foldNet = netopt(foldNet, options, trainExamples, trainLabels1ofN, method);
        
        % apply to test set
        
        yt = mlpfwd(foldNet, testExamples);
                
        [yvalue,ypos]   = max(yt,[],2);
        predictedLabels = sortedLabels(ypos);
        errorAfterBurst{k}(b) = sum(predictedLabels~=testLabels)/length(testLabels);
    end
    
    [errorPerFold(k),nBurstsPerFold(k)] = min(errorAfterBurst{k});
end

nIter = ceil(median(nBurstsPerFold))*nIterPerBurst;

