% do once
clear all
load childru_data
load best_preps_params
curdir=pwd;
bss='/'; % change to '\' for Windows
addpath([curdir bss 'classifier']);
addpath([curdir bss 'utility']);

%% Arrange the temp data for training and testing
clear mycell_temp
isval=0;%set 1 for cross-validation (optimizing params), 0 for testing with optimized hyper params
targetID=3;% Set 1 for Age, 2 for Gender, and 3 for Emotion
% EmotionIDs -> 1: Comfort, 2: Discomfort, 3: Neutral

if isval
    
    trInd=childru_data{2, 1}.metadata(:,4)<45;
    valInd=~trInd;
    mycell_temp{1}.traindata=childru_data{2, 1}.data(trInd,:);
    mycell_temp{1}.testdata=childru_data{2, 1}.data(valInd,:);

    mycell_temp{1}.trainlabels=childru_data{2, 1}.metadata(trInd,targetID);
    mycell_temp{1}.testlabels=childru_data{2, 1}.metadata(valInd,targetID);

    mycell_temp{2}.traindata=mycell_temp{1}.testdata;
    mycell_temp{2}.testdata=mycell_temp{1}.traindata;
    mycell_temp{2}.trainlabels=mycell_temp{1}.testlabels;
    mycell_temp{2}.testlabels=mycell_temp{1}.trainlabels;
else
    mycell_temp{1}.traindata=childru_data{2, 1}.data;
    mycell_temp{1}.testdata=childru_data{1, 1}.data;
    mycell_temp{1}.testlabels=childru_data{1, 1}.metadata(:,targetID);
    mycell_temp{1}.trainlabels=childru_data{2, 1}.metadata(:,targetID);
end

disp('temp data loaded')
%% Compute Kernels
norm_type=1;%1:min-max, 2: z-norm
kernel_type=1;%1 for linear, 2 for RBF
if ~isval % if test then use optimized preprocessing options
    norm_type=best_prep(targetID,1);
    kernel_type=best_prep(targetID,2);
end
dim = size(mycell_temp{1}.traindata,2);
gamma=1/dim;

do_filter=1;

tiny=1e-4;
std_filter_all=std(mycell_temp{1,1}.traindata)>eps;
for i=1:numel(mycell_temp)
   
    temp_trdata=mycell_temp{i}.traindata;
    temp_numdata=mycell_temp{i}.testdata;


      if (do_filter)  
        std_filter=std_filter_all;

        temp_trdata=temp_trdata(:,std_filter);
        temp_numdata=temp_numdata(:,std_filter);
  
      end
     if (norm_type==1)
         [temp_trdata,setting]=mapminmax(temp_trdata');
         [temp_numdata]=mapminmax('apply',temp_numdata',setting);
         temp_trdata=temp_trdata';
         temp_numdata=temp_numdata';

     elseif (norm_type==2)
         [temp_trdata,mx,stdx]=autosc(temp_trdata);
         [temp_numdata]=scal(temp_numdata,mx,stdx);
     end
  
    mycell_temp{i}.trndata=temp_trdata;
    
    mycell_temp{i}.valdata=temp_numdata;
    
    if (kernel_type==1)
        mycell_temp{i}.train_kernel=(temp_trdata*temp_trdata');
        mycell_temp{i}.val_kernel=(temp_numdata*temp_trdata');
    else 
        mycell_temp{i}.train_kernel=exp(-gamma*dist(temp_trdata,temp_trdata').^2);
        mycell_temp{i}.val_kernel=exp(-gamma*dist(temp_numdata,temp_trdata').^2);
    end
    
end
disp('kernels computed');
%% classify the struct
if isval
    nComp_set=2:2:24;% the length of hyper param sets must be equal
    C_set_elm=10.^[-6:5]; % with training/val set
    C_set_svm=C_set_elm;
   [perf_three_methods_val,best_perf_val,best_pars,all_preds_val,mycell_temp] = classify_struct(mycell_temp,kernel_type,nComp_set,C_set_elm,C_set_svm);
    best_params(targetID,:)=best_pars;
else
    nComp_set=best_params(targetID,1); 
    C_set_elm=best_params(targetID,2); 
    C_set_svm=best_params(targetID,3); 

    [perf_three_methods,best_perf_test,~,all_preds_test] = classify_struct(mycell_temp,kernel_type,nComp_set,C_set_elm,C_set_svm);
end
%% Further breakdown analysis
% For each dimension it is possible to carry out confusion matrix analysis 
% an example is given for emotions with age group breakdown
% assuming tests are done and emotion preds are in all_preds_test
preds_elm=all_preds_test(:,2);
age_groups= childru_data{1, 1}.metadata(:,1);


gt_emo_ag1=childru_data{1, 1}.metadata(age_groups==1,3); % age group 3-4
gt_emo_ag2=childru_data{1, 1}.metadata(age_groups==2,3); % age 5
gt_emo_ag3=childru_data{1, 1}.metadata(age_groups==3,3); % age group 6-7

preds_ag1=preds_elm(age_groups==1);
preds_ag2=preds_elm(age_groups==2);
preds_ag3=preds_elm(age_groups==3);

[UAR_ag1,Accuracy_ag1,Recall_ag1,CM_ag1] = getUAR(gt_emo_ag1,preds_ag1);
[UAR_ag2,Accuracy_ag2,Recall_ag2,CM_ag2] = getUAR(gt_emo_ag2,preds_ag2);
[UAR_ag3,Accuracy_ag3,Recall_ag3,CM_ag3] = getUAR(gt_emo_ag3,preds_ag3);
% row normalize confusion matrices to reproduce results on Table 9
class_list= childru_data{1, 1}.classlist
CM_ag1_rn=CM_ag1./repmat(sum(CM_ag1,2),1,size(CM_ag1,2))
CM_ag2_rn=CM_ag2./repmat(sum(CM_ag2,2),1,size(CM_ag2,2))
CM_ag3_rn=CM_ag3./repmat(sum(CM_ag3,2),1,size(CM_ag3,2))

% to reproduce Table 9 we need to permute CM
permut=[2 3 1]; %  comf disc neut  -> disc neut comf
CM_ag1_rn=CM_ag1_rn(permut,permut);
CM_ag2_rn=CM_ag2_rn(permut,permut);
CM_ag3_rn=CM_ag3_rn(permut,permut);
Table9=round(100*[CM_ag1_rn CM_ag2_rn CM_ag3_rn])
