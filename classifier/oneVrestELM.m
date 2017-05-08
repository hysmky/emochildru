function [predict_label,accuracy,dec_values] = oneVrestELM(traindata,trainlabel,testdata,testlabel,C)
% ====================================================================================
% 
% 
% Author:Heysem Kaya @ BU/CmpE (alternative for Mengyi Liu's one vs rest
% PLS)
% E-mail: heysem@boun.edu.tr
% 
% Nov.29, 2014
% ====================================================================================
dec_values = zeros(length(testlabel),length(unique(trainlabel)));
Ntr=size(traindata,1);
for i = 1:length(unique(trainlabel))
    binarytrainlabel = zeros(length(trainlabel),1);
    idxtrain = (trainlabel==i);
    binarytrainlabel(idxtrain) = 1;
    
    beta = (traindata+eye(Ntr)/C)\binarytrainlabel;
    
    dec_values(:,i) = (testdata)*beta;
end
[~,predict_label] = max(dec_values,[],2);
accuracy = length(find(predict_label==testlabel))/length(testlabel);
fprintf('Accuracy = %f\n',accuracy);
