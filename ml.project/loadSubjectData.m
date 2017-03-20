function [examples, labels] = loadSubjectData(datafile)


    load(datafile)  
   
    % collect the non-noise and non-fixation trials
    trials = find([info.cond] > 1); 
    [info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);
    
    roisOfInterest = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
    [info2,data2,meta2] = transformIDM_selectROIVoxels(info1,data1,meta1, roisOfInterest);

    % seperate P1st and S1st trials
    [infoP1,dataP1,metaP1] = transformIDM_selectTrials(info2,data2,meta2,find([info1.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1] = transformIDM_selectTrials(info2,data2,meta2,find([info1.firstStimulus]=='S'));
 
    % seperate reading P vs S
    [infoP2,dataP2,metaP2] = transformIDM_selectTimewindow(infoP1,dataP1,metaP1,(1:16));
    [infoP3,dataP3,metaP3] = transformIDM_selectTimewindow(infoS1,dataS1,metaS1,(17:32));
    [infoS2,dataS2,metaS2] = transformIDM_selectTimewindow(infoP1,dataP1,metaP1,(17:32));
    [infoS3,dataS3,metaS3] = transformIDM_selectTimewindow(infoS1,dataS1,metaS1,(1:16));

    % convert to examples
    [examplesP2,labelsP2,exInfoP2] = idmToExamples_condLabel(infoP2,dataP2,metaP2);
    [examplesP3,labelsP3,exInfoP3] = idmToExamples_condLabel(infoP3,dataP3,metaP3);
    [examplesS2,labelsS2,exInfoS2] = idmToExamples_condLabel(infoS2,dataS2,metaS2);
    [examplesS3,labelsS3,exInfoS3] = idmToExamples_condLabel(infoS3,dataS3,metaS3);

    % combine examples and create labels.  Label 'picture' 1, label 'sentence' 2.
    examplesP = [examplesP2;examplesP3];
    examplesS = [examplesS2;examplesS3];
    labelsP = ones(size(examplesP,1),1);
    labelsS = ones(size(examplesS,1),1)+1;
    examples = [examplesP;examplesS];
    labels = [labelsP;labelsS];
    
end