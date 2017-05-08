function [predict_label,accuracy,dec_values] = oneVrestSVM(traindata,trainlabel,testdata,testlabel,C,kernel_type,gamma)
% ====================================================================================
% 
% 
% Author:Heysem Kaya @ BU/CmpE 
% 
% E-mail: heysem@boun.edu.tr
% 
% Apr.9, 2015
% ====================================================================================
if nargin<6
    kernel_type=1;
    
end
dec_values = zeros(size(testlabel,1),length(unique(trainlabel)));
Ntr=size(traindata,1);
for i = 1:length(unique(trainlabel))
    binarytrainlabel = zeros(length(trainlabel),1);
    idxtrain = (trainlabel==i);
    binarytrainlabel(idxtrain) = 1;
    
    %beta = (traindata+eye(Ntr)/C)\binarytrainlabel;
    if (kernel_type==1)
        SVMModel=fitcsvm(traindata,binarytrainlabel,'BoxConstraint', C,'KernelFunction','linear','ClassNames',[0 1]);
    else
        if nargin==7
            SVMModel=fitcsvm(traindata,binarytrainlabel,'BoxConstraint', C,'KernelFunction','RBF','KernelScale',1/gamma,'ClassNames',[0 1]);
        else
            SVMModel=fitcsvm(traindata,binarytrainlabel,'BoxConstraint', C,'KernelFunction','RBF','KernelScale','auto','ClassNames',[0 1]);
        end
    end
    [~,scores]=predict(SVMModel,testdata);
    dec_values(:,i) = scores(:,2);
end
[~,predict_label] = max(dec_values,[],2);
accuracy = length(find(predict_label==testlabel))/length(testlabel);
fprintf('Accuracy SVM = %f\n',accuracy);
