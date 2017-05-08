function [best_perf,best_models]=eval_two_methods(train_kernel,trainlabels,val_kernel,vallabels,C_set,nComp,do_elm,do_pls)
    if nargin<6
        nComp=20;
    end
        
    if nargin < 5
        C_set=1;
    end
    best_perf=zeros(1,2);

    if (do_pls)
        
        best_score_pls=[];
        best_uar_pls=0;
        bestComp_pls=0;
        for ncomp =nComp
            [pred_pls,accur_pls,score_pls] = oneVrestPLS(train_kernel,trainlabels,val_kernel,vallabels,ncomp);
            
            [uar_pls]=getUAR(vallabels,pred_pls);
            
            if (best_uar_pls<uar_pls)
                best_uar_pls=uar_pls;
                best_score_pls=score_pls';
                bestComp_pls=ncomp;
            end
        end


        best_models.best_score_pls=best_score_pls;
        [~,best_models.best_pred_pls]=max(best_score_pls);
        [best_models.best_UAR_pls,best_models.best_accuracy_pls]=...
            getUAR(vallabels,best_models.best_pred_pls');
       
        best_models.bestComp_pls=bestComp_pls;
        best_perf(1,1)=best_uar_pls;
   
    end
    
    if do_elm 
        %C_set=2.^[-15:15]; % 
        accuracy=zeros(size(C_set));
        best_score_elm=[];
        best_uar_elm=0;
        bestC_elm=0;
        j=0;
        for C=C_set
            j=j+1;
            [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,Y,TY] = elm_kern(train_kernel, trainlabels, val_kernel, vallabels, 1, C);
            accuracy(j)=TestingAccuracy;
            [~,pred_pls]=max(TY);
            
            [uar_pls]=getUAR(vallabels,pred_pls');
            
            if (best_uar_elm<uar_pls)
                best_uar_elm=uar_pls;
                best_score_elm=TY;
                bestC_elm=C;
            end
        end
        best_models.best_score_elm=best_score_elm;
        [~,best_models.best_pred_elm]=max(best_score_elm);
        [best_models.best_UAR_elm, best_models.best_accuracy_elm]=...
            getUAR(vallabels,best_models.best_pred_elm');
      
        best_models.bestC_elm=bestC_elm;

        best_perf(1,2)=best_uar_elm;
        %best_models.best_UAR_elm;
    end
end
 
