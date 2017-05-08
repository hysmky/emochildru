function [predict_label,accuracy,dec_values] = oneVrestPLS(traindata,trainlabel,testdata,testlabel,ncomp)
% ====================================================================================
% The released code for EmotiW 2014 challenge
% 
% Author: Mengyi Liu @ VIPL,ICT,CAS
% E-mail: mengyi.liu@vipl.ict.ac.cn
% 
% Nov.4, 2014
% ====================================================================================
dec_values = zeros(length(testlabel),length(unique(trainlabel)));
for i = 1:length(unique(trainlabel))
    binarytrainlabel = zeros(length(trainlabel),1);
    idxtrain = (trainlabel==i);
    binarytrainlabel(idxtrain) = 1;

    [~,~,~,~,beta] = plsregress(traindata,binarytrainlabel,ncomp);
    dec_values(:,i) = [ones(size(testdata,1),1) testdata]*beta;
end
[~,predict_label] = max(dec_values,[],2);
accuracy = length(find(predict_label==testlabel))/length(testlabel);
fprintf('Accuracy PLS = %f\n',accuracy);
