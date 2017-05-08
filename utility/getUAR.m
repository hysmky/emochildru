function [ UAR,Accuracy,Recall,CM ] = getUAR( targets,preds )
%GETUAR Summary of this function goes here
%   Detailed explanation goes here
CM=confusionmat(targets,preds);
C2=sum(CM,2);
Accuracy=sum(diag(CM))/numel(targets);
if (sum(C2==0))
   disp(['some classes have no instances']); 
end
Recall=diag(CM)./C2;
UAR=mean(Recall);

end

