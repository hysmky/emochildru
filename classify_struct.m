function [perf_three_methods,max_perf,best_params,all_preds,mycell_temp] = classify_struct(mycell_temp,kernel_type,nComp_set,C_set_elm,C_set_svm)

tS=numel(nComp_set);
do_elm=1;
do_pls=1;
do_svm=1;
perf_three_methods=zeros(tS,3); 
Nfolds=numel(mycell_temp);
for l=1:tS
    for i=1:Nfolds
        [mycell_temp{i}.best_linperf,mycell_temp{i}.best_linmodels]=...
            eval_two_methods(mycell_temp{i}.train_kernel,mycell_temp{i}.trainlabels,...
            mycell_temp{i}.val_kernel,mycell_temp{i}.testlabels,C_set_elm(l),nComp_set(l),do_elm,do_pls);    
        
    if do_svm 
       

        best_score_svm=[];
        best_uar_svm=0;
        bestC_svm=0;
       
         [pred_svm,accur_svm,score_svm] = oneVrestSVM(mycell_temp{i}.trndata,...
             mycell_temp{i}.trainlabels,...
             mycell_temp{i}.valdata,...
             mycell_temp{i}.testlabels,C_set_svm(l),kernel_type);
            
            [uar_svm]=getUAR(mycell_temp{i}.testlabels,pred_svm);
            
        if (best_uar_svm<uar_svm)
            best_uar_svm=uar_svm;
            best_score_svm=score_svm';
            bestC_svm=C_set_elm(l);
        end
        
        mycell_temp{i}.best_linmodels.best_score_svm=best_score_svm;
        [~,mycell_temp{i}.best_linmodels.best_pred_svm]=max(best_score_svm);
      
        mycell_temp{i}.best_linmodels.best_accuracy_svm=best_uar_svm;
        mycell_temp{i}.best_linmodels.bestC_svm=bestC_svm;

        mycell_temp{i}.best_linperf(1,3)= best_uar_svm;
        %best_models.best_UAR_elm;
    end
        
    end
    
    % compute overall UAR
   
    gt=[];
    preds_elm=[];
    preds_pls=[];
    preds_svm=[];
    
    for i=1:Nfolds
        gt=[gt;mycell_temp{i}.testlabels];
        if do_elm
            preds_elm=[preds_elm;mycell_temp{i}.best_linmodels.best_pred_elm'];
        end
        if do_pls
            preds_pls=[preds_pls;mycell_temp{i}.best_linmodels.best_pred_pls'];
        end
        
        if do_svm
            preds_svm=[preds_svm;mycell_temp{i}.best_linmodels.best_pred_svm'];
        end
    end
    if do_pls
        [UAR_pls,Acc_pls,Recall_pls,CM_pls]=getUAR(gt,preds_pls);
        perf_three_methods(l,1)=UAR_pls;
    end
    if do_elm
        [UAR_elm,Acc_elm,Recall_elm,CM_elm]=getUAR(gt,preds_elm);
        perf_three_methods(l,2)=UAR_elm;
    end
    
    if do_svm
        [UAR_svm,Acc_svm,Recall_svm,CM_svm]=getUAR(gt,preds_svm);
        perf_three_methods(l,3)=UAR_svm;
    end

end
[max_perf,max_ind]=max(perf_three_methods,[],1);
max_perf
best_params=[nComp_set(max_ind(1)) C_set_elm(max_ind(2)) C_set_svm(max_ind(3))];
perf_three_methods=[nComp_set' perf_three_methods C_set_elm' C_set_svm'];
all_preds=[preds_pls preds_elm preds_svm];
end