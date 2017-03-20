
% load the data
[examples, labels] = loadSubjectData('data-starplus-04847-v7.mat');

% figure out a few things
sortedLabelValues = sort(unique(labels));
nClasses = length(sortedLabelValues);
[numTrain, numFeatures] = size(examples);


% Train the classifier
% uses the NETLAB code by Ian Nabney

models = cell(nClasses+3,1);

classifierParameters = {}; %{20,20,30};

net = classifierNeuralNetwork(examples,labels,classifierParameters);

% store model in the "discriminative" part
models{nClasses+1} = net;


% Training Set information
trainingSetInfo.classifierParameters = classifierParameters;
trainingSetInfo.nExamples            = numTrain;
trainingSetInfo.nFeatures            = numFeatures;
trainingSetInfo.nClasses             = nClasses;
trainingSetInfo.sortedLabelValues    = sortedLabelValues;
trainingSetInfo.classPriors          = zeros(nClasses,1);

for l=1:nClasses
    trainingSetInfo.classPriors(l) = length(find(labels==sortedLabelValues(l)));
end

trainingSetInfo.classPriors = trainingSetInfo.classPriors/numTrain;

% restore to make sure
models{nClasses+2} = trainingSetInfo; % kept for compatibility


% Classifier structure
trainedClassifier.models          = models;
trainedClassifier.trainingSetInfo = trainingSetInfo;
trainedClassifier.classifier      = 'neural';
trainedClassifier.classifierParameters = classifierParameters;

%Test

% load test data
[testExamples, testLabels] = loadSubjectTestData('data-starplus-04847-v7.mat');

% normalize the data to mean 0 std 1
testExamples = normalize(testExamples); 

% measure performance with test data
scores = mlpfwd(net,testExamples);

[result,predictedLabels,trace] = summarizePredictions(scores,trainedClassifier,'accuracy',testLabels);
