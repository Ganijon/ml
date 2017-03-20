  
%
% Ancillary code 
%

%% Transform a list of labels into a "1 of N" encoding
%% (a binary matrix with as many rows as examples, and as many
%% columns as labels. The row for an example is 1 in the column
%% corresponding to its label, and 0 everywhere else. The columns
%% are in the order of the labels sorted by value.

function [labels1ofN] = oneOfNencoding(labels)

classes   = unique(labels); nClasses = length(classes);
nExamples = length(labels);

labels1ofN = zeros(nExamples,nClasses);
for c = 1:nClasses
  label           = classes(c);
  labels1ofN(:,c) = (labels == label);
end

