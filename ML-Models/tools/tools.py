import sys,os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import time
from sklearn.metrics import confusion_matrix
from skmultilearn.model_selection import iterative_train_test_split 
from collections import Counter
from scipy import integrate
import random
import string
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from IPython.display import display, clear_output
from tqdm import tqdm
import ipywidgets as widgets


class tools():
    def __init__(self):
        self.__zero=1E-9
        self.__numDecs=3
        self.__pTh=[0.5,0.5]
        self.__numberOfClasses=2
        self.__classesNames=["",""]
        self.__cfMtx=pd.DataFrame()
        return


    def joinContingenceMatrices(self,cfMtxs,classifierslabels):
        allMtxs=pd.DataFrame()
        for i,mtx_ in enumerate(cfMtxs): 
            mtx_=mtx_
            mtx_["label"]=classifierslabels[i]
            allMtxs=pd.concat([allMtxs,mtx_],sort=False,ignore_index=False)
        return allMtxs

    
    def kFoldsRegressorSplitWithStratification(self,idxs,stratificationLabels,k_folds=10,test_size=0.2,rand_seed=42):
        np.random.seed(rand_seed)
        random.seed(rand_seed)

        if k_folds>1:
            seeds_=random.sample(range(k_folds*100), k=k_folds)
            for k_fold,int_rand_seed in enumerate(seeds_):
                np.random.seed(int_rand_seed)
                idxsRandomized=np.random.choice(range(0,len(idxs)), size=len(idxs), replace=False)
                X_train, X_test = train_test_split(
                    idxs, 
                    test_size=test_size,
                    stratify=stratificationLabels,
                    random_state=int_rand_seed,
                    shuffle=True
                )
                yield (list(X_train),list(X_test))
        else:
            return


    def getTestRegressorMetricsStatistics(self,inputModel,X_test,Y_test,Y_stratification=None,n_folds=10,partitionFraction=0.5,rand_seed=58):

        np.random.seed(rand_seed)
        randSeeds=np.random.choice(range(0,10000), size=int(n_folds/2), replace=False)
        i=0
        metrics=pd.DataFrame()
        for l,seed_ in enumerate(randSeeds):
            set_idx=range(0,len(Y_test))
            kfold_A_idx, kfold_B_idx = train_test_split(
                set_idx,
                test_size=partitionFraction,
                stratify=Y_stratification,
                random_state=seed_,
                shuffle=not(isinstance(Y_stratification,type(None)))
            )

            for kfold_idxs in (kfold_A_idx, kfold_B_idx):
                i+=1
                y_kfold=Y_test[kfold_idxs]
                X_kfold=X_test[kfold_idxs]
                y_pred_regression_kfold=inputModel.predict(X_kfold)
                mae=mean_absolute_error(y_kfold,y_pred_regression_kfold)
                rmse=np.sqrt(mean_squared_error(y_kfold,y_pred_regression_kfold))
                r2=r2_score(y_kfold,y_pred_regression_kfold)
                metrics=pd.concat([metrics,pd.DataFrame({"kfold":[i],"MAE":[mae],"RMSE":[rmse],"R2":[r2]})],sort=False,ignore_index=True) 

        vmetrics =metrics.agg(["mean","std",self.getUncertainty])
        vmetrics = vmetrics.T.reset_index(drop=False).rename(columns={"index":"score","getUncertainty":"U.95"})[1:]
        vmetrics['n'] = len(metrics)
        vmetrics['data']=None
        vmetrics=vmetrics.set_index("score",drop=False)
        vmetrics.at['MAE', 'data']=metrics.MAE.to_list()
        vmetrics.at['RMSE', 'data']=metrics.RMSE.to_list()
        vmetrics.at['R2', 'data']=metrics.R2.to_list()
        vmetrics=vmetrics.reset_index(drop=True)

        return vmetrics

    def getUncertainty(self,values,pconf=0.95,distribution="t.student"):
        n_=len(values)

        if distribution=="normal":
            k=stats.norm.interval(0.95)[1]
        elif distribution=="t.student":
            k=stats.t.interval(0.95,n_-1)[1]
        else:
            k=stats.t.interval(0.95,n_-1)[1]

        U=np.std(values,ddof=1)/np.sqrt(n_)*k
        return  U


    def getRandomSet(self,exploring_data_idxs,randomState=40): 
        random.seed(randomState)
        trainingSetDF=pd.DataFrame()
        for grp in exploring_data_idxs.groupby(['LogKow_categories'],observed=False):
            sampling_list=grp[1].copy().idxs.to_list()
            category_=grp[1].LogKow_categories.iloc[0]
            for i in range(0,4):
                currRandomSample=random.sample(sampling_list,12)
                trainingSetDF=pd.concat([trainingSetDF,pd.DataFrame({'LogKow_category':[category_],
                                                                     'internalGroup':[i],
                                                                     'elements':[currRandomSample]
                                                                    })],sort=False,ignore_index=True)
                sampling_list=list(set(sampling_list) -set(currRandomSample))

            trainingSetDF=pd.concat([trainingSetDF,pd.DataFrame({'LogKow_category':[category_],
                                                                 'internalGroup':[i+1],
                                                                 'elements':[sampling_list]
                                                                })],sort=False,ignore_index=True)


        return trainingSetDF

    def getTrainingTestSetsRandom(self,exploring_data_idxs,numSets=1,randomState=42):
        sampledTestSet=self.getRandomSet(exploring_data_idxs,randomState=randomState)
        for i in range(0,numSets):
            random.seed(i)
            selection_=random.choices(range(0,5),k=10)
            trainingIndexes=[]
            testingIndexes=[]
            for idx,grp in enumerate(sampledTestSet.groupby(['LogKow_category'],observed=False)):
                testingIndexes=testingIndexes+grp[1][grp[1].internalGroup==selection_[idx]].elements.to_list()[0]
            trainingIndexes=list(set(exploring_data_idxs.idxs)-set(testingIndexes))
            yield(trainingIndexes,testingIndexes)


    def assessmentResults(self,y_exp,y_predictions,nAssess):
        tR_results=pd.DataFrame({'tR_exp':y_exp,
                                 'tR_pred':y_predictions,
                                 'error':y_predictions-y_exp,
                                }).sort_values('tR_pred')
        tR_results.insert(0,'nAssess',nAssess+1)
        tR_results=tR_results.reset_index(drop=True)
        return tR_results
    

    def getMatthewsCorrelationCoefficient(self,cfxMtx,transpose=True,dZ=1E-10):
        wcfxMtx=cfxMtx.copy()
        if transpose: wcfxMtx=cfxMtx.transpose().copy()
        ts=wcfxMtx.sum(axis=0).to_list()
        ps=wcfxMtx.sum(axis=1).to_list()
        s=wcfxMtx.sum().sum()
        c=np.diag(wcfxMtx.values).sum()
        MCC=(c*s-np.dot(ps,ts))/(np.sqrt( ( np.dot(s,s)-np.dot(ps,ps))*( np.dot(s,s)-np.dot(ts,ts)) )+dZ)
        return MCC
    
    def getSingleLabelMetrics(self,singleLabelCfMtx,beta=1,pDg=3,dZ=1E-10):
        beta2=beta**2
        classNames=singleLabelCfMtx.columns.to_list()
        labelMetrics=pd.DataFrame()
        accuracy_=np.round(np.diag(singleLabelCfMtx).sum()/(singleLabelCfMtx.sum().sum()+dZ)*100,pDg)
        labelMetrics[f"Accuracy"]=[accuracy_]
        for className_ in np.array(classNames).astype(str):            
            recall_=np.round(singleLabelCfMtx[className_].loc[className_]/(singleLabelCfMtx[className_].sum()+dZ)*100,pDg)
            labelMetrics[f"Recall({className_})"]=[recall_]
            precision_=np.round(singleLabelCfMtx.loc[className_][className_]/(singleLabelCfMtx.loc[className_].sum()+dZ)*100,pDg)
            labelMetrics[f"Precision({className_})"]=[precision_]
            labelMetrics[f"F_{beta}({className_})"]=[np.round((1+beta2)*(recall_*precision_)/(recall_+(beta2)*precision_+dZ),pDg)]

        combinatedMetrics=pd.DataFrame()
        _row=labelMetrics.iloc[0]    
        labelMetrics.loc[0,'MCC']=[self.getMatthewsCorrelationCoefficient(singleLabelCfMtx)]
        return labelMetrics

    def getSingleLabelContingenceMatrix(self,predProb,y_target,classNames=None,pThres=None):
        y_target_=y_target.astype(int)
        if isinstance(classNames,type(None)):
            self.__classesNames=np.sort(np.unique(y_target.astype(int))).astype(str)
        else:
            self.__classesNames=np.array(classNames).astype(str)

        self.__numberOfClasses=len(self.__classesNames)
        if isinstance(pThres,type(None)):
            self.__pTh=['max']*self.__numberOfClasses
            if self.__numberOfClasses==2:self.__pTh=[0.5,0.5]
            predictionsLabs=np.array([list(instance>=max(instance)) for instance in predProb])
        else:
            self.__pTh=pThres
            if not(isinstance(pThres,list)):
                if len(self.__classesNames)==2:
                    self.__pTh=[pThres,1.0-pThres]    
                else:
                    self.__pTh=[pThres]*len(self.__classesNames)
            predictionsLabs=np.array(list(map(lambda i:list(predProb[:,i]>=(self.__pTh[i])),range(np.shape(predProb)[1])))).transpose()

        predictionsLabs=predictionsLabs.astype(str)
        predictionsLabs=np.char.replace(predictionsLabs, 'False', '')

        cfMtx=pd.DataFrame()
        for i,class_ in enumerate(self.__classesNames):
            predictionsLabs[:,i]=np.char.replace(predictionsLabs[:,i],'True',class_)
        predictionsLabs=[''.join(row) for row in predictionsLabs]

        cfMtx=pd.DataFrame()
        for idx,class_ in enumerate(self.__classesNames):
            clabels=np.array(np.array(y_target_).astype(str))==class_
            classPredictions=pd.DataFrame(np.unique(np.array(predictionsLabs)[clabels],return_counts=True))

            classPredictions.columns=classPredictions.iloc[0]
            classPredictions=classPredictions.drop(0,axis=0).reset_index(drop=True)
            overlapedClasses=list(set(classPredictions.columns)-set(self.__classesNames))

            emptyClasses=list(set(self.__classesNames)-set(classPredictions.columns))
            emptyClasses = classPredictions.columns.tolist() + emptyClasses
            classPredictions = classPredictions.reindex(columns=emptyClasses)
            classPredictions=classPredictions.replace(np.nan, 0)
            for oClass in overlapedClasses:
                if class_ in oClass: classPredictions.loc[0,class_]=classPredictions[class_].iloc[0]+classPredictions.loc[0,oClass]
            classPredictions=classPredictions[self.__classesNames]


            classPredictions['target']=class_
            cfMtx=pd.concat([cfMtx,classPredictions])    

        cfMtx=cfMtx.sort_index(axis=1).set_index('target')
        cfMtx.columns.name='prediction'
        cfMtx=cfMtx.transpose()
        self.__cfMtx=cfMtx
        return cfMtx


    def getMetricsAtProbabilityThreshold(self,predictionsProb,y_target,labels=[],probSteps=20,pThres=None):
        compiled_metrics=pd.DataFrame()
        probsShape=np.shape(predictionsProb)
        if(len(probsShape)==2):
            predictionsProb_=np.reshape(predictionsProb,(1,len(predictionsProb),probsShape[1]))
        elif( len(probsShape)==3):
            predictionsProb_=predictionsProb
        else:
            return

        probsShape=np.shape(predictionsProb_)
        labelsNum=probsShape[0]

        _labels=labels
        if ( (len(labels)!=labelsNum) | (len(labels)==0) ):
            _labels=list(np.array(range(labelsNum)).astype(str))
            
        y_targetT=y_target.copy()
        if ( (labelsNum==1) &  (len(np.shape(y_targetT))==1 ) ):
            y_targetT=np.reshape(y_targetT,(len(y_targetT),1))

        if np.shape(y_targetT)[1]!=labelsNum:
            return


        for idxLabel in range(labelsNum):
            y_target_=y_targetT[:,idxLabel].astype(int)
            classNames=list(np.sort(np.unique(y_target_))) 
            label_probs=predictionsProb_[idxLabel]
            numInstantes_=len(y_target_)

            if isinstance(pThres,float):
                cfMtxSL=self.getSingleLabelContingenceMatrix(label_probs,y_target_,classNames,pThres)
                metrics_=self.getSingleLabelMetrics(cfMtxSL)
                for clss,pth_ in zip(self.__classesNames,self.__pTh):
                    metrics_[f"Prob({clss})"]=[pth_]
                metrics_['label']=[_labels[idxLabel]]
                compiled_metrics=pd.concat([compiled_metrics,metrics_],sort=False,ignore_index=True)

            else:
                for prob_step in range(0,probSteps+1):
                    prob_threshold=1.0-prob_step/probSteps
                    cfMtxSL=self.getSingleLabelContingenceMatrix(label_probs,y_target_,classNames,prob_threshold)
                    metrics_=self.getSingleLabelMetrics(cfMtxSL)
                    for clss,pth_ in zip(self.__classesNames,self.__pTh):
                        metrics_[f"Prob({clss})"]=[pth_]
                    metrics_['label']=[_labels[idxLabel]]
                    compiled_metrics=pd.concat([compiled_metrics,metrics_],sort=False,ignore_index=True)

        return compiled_metrics.reset_index(drop=True)


    def getRamdomClassificationModel(self,classes=2,Y_target=None,vsize=500,rand_seed=None):
        np.random.seed(rand_seed)
        random.seed(rand_seed)
        
        _vsize=vsize
        if isinstance(Y_target,type(None)):
            if isinstance(classes,list):
                _classes=classes
            elif isinstance(classes,int):
                _classes=list(range(classes))
            else:
                return ()
            y_target_=[(random.sample(_classes, k=1))[0] for i in range(_vsize)]
        else:
            _classes=list(np.unique(Y_target))
            y_target_=Y_target.astype(int)
            _vsize=len(Y_target)
            
        randomProbs_=np.random.rand(_vsize, len(_classes))
        randomProbs_=np.array([list(randomProbs_[i,:]/np.sum(randomProbs_[i,:])) for i in range(len(randomProbs_))]) 
        return (randomProbs_,y_target_)
    

    def getAveragePRC(self,statData):
        PRCs=statData.iloc[0].PRC_curves.copy()

        if "label" in PRCs.columns:
            dataLabels=np.unique(PRCs["label"])
        else:
            PRCs["label"]="-"
            dataLabels=["-"]

        allMetricsAverages=pd.DataFrame()
        for label_ in dataLabels:
            PRCs_=PRCs[PRCs["label"]==label_].copy().drop(columns=["label"]).reset_index(drop=True)
            
            PRCs_=PRCs_[~((PRCs_.Recall==1.0) & (PRCs_.Precision==0.0))]
            PRCs_=PRCs_.groupby("Prob").agg(['mean','std',self.getUncertainty,'count']).drop(columns="kfold").reset_index(drop=False)
            PRCs_ = PRCs_.T.reset_index(drop=False).T
            cols=list(map(lambda x:x[0]+"_"+x[1] ,zip(PRCs_.loc["level_0"],PRCs_.loc["level_1"])))
            cols[0]="Prob"
            PRCs_.columns=cols
            PRCs_=PRCs_.iloc[2:,:]
            PRCs_.columns=[cname.replace("getUncertainty", "U").replace("count", "n") for cname in PRCs_.columns]
            PRCs_["label"]=label_
            allMetricsAverages=pd.concat([allMetricsAverages,PRCs_],sort=False,ignore_index=True)

        return allMetricsAverages
    
    def getAUCsAverages(self,statData):
        AUCs=statData.iloc[0].AUCs.copy()

        if "label" in AUCs.columns:
            dataLabels=np.unique(AUCs["label"])
        else:
            AUCs["label"]="-"
            dataLabels=["-"]

        allMetricsAverages=pd.DataFrame()
        for label_ in dataLabels:
            AUCs_=AUCs[AUCs["label"]==label_].copy().drop(columns=["label"]).reset_index(drop=True)
            AUCs_=AUCs_.agg(['mean','std',self.getUncertainty,'count']).drop(columns="kfold").reset_index(drop=False)
            AUCs_=AUCs_.T
            AUCs_.columns=AUCs_.loc['index']
            AUCs_=AUCs_.iloc[1:]
            AUCs_.columns=[cname.replace("getUncertainty", "U").replace("count", "n") for cname in AUCs_.columns]
            AUCs_["label"]=label_
            allMetricsAverages=pd.concat([allMetricsAverages,AUCs_],sort=False,ignore_index=False)

        allMetricsAverages=allMetricsAverages.reset_index(drop=False)
        allMetricsAverages=allMetricsAverages.rename(columns={"index":"score"})
        return allMetricsAverages

    
    def getMetricsAverages(self,statData,valsType="maxValsMetrics"):
        if valsType!="ovsOthMetrics": 
            metricsAverages=statData.iloc[0][valsType].copy()        
            if "label" in metricsAverages.columns:
                dataLabels=np.unique(metricsAverages["label"])
            else:
                metricsAverages["label"]="-"
                dataLabels=["-"]

            allMetricsAverages=pd.DataFrame()
            for label_ in dataLabels:
                metricsAverages_=metricsAverages[metricsAverages["label"]==label_].copy().drop(columns=["label"]).reset_index(drop=True)
                metricsAverages_=metricsAverages_.agg(['mean','std',self.getUncertainty,'count']).drop(columns="kfold").reset_index(drop=False)
                metricsAverages_=metricsAverages_.T
                metricsAverages_.columns=metricsAverages_.loc['index']
                metricsAverages_=metricsAverages_.iloc[1:]
                metricsAverages_.columns=[cname.replace("getUncertainty", "U").replace("count", "n") for cname in metricsAverages_.columns]
                metricsAverages_["label"]=label_
                allMetricsAverages=pd.concat([allMetricsAverages,metricsAverages_],sort=False,ignore_index=False)

            allMetricsAverages=allMetricsAverages.reset_index(drop=False)
            allMetricsAverages=allMetricsAverages.rename(columns={"index":"score"})
        else:
            metricsAverages=statData.iloc[0][valsType].copy()        
            allMetricsAverages=self.getAverageOvsOth(metricsAverages)
            
        return allMetricsAverages


    
    def getIMCPCurves(self,predictionsProb,y_target,classifierslabels=None):

        predictionsProb_=predictionsProb.copy()
        y_target_=y_target.copy()
        dim_=len(np.shape(y_target_))
        if dim_==1:
            y_target_=np.reshape(y_target_,(len(y_target_),dim_))
            predictionsProb_=np.reshape(predictionsProb_,(dim_,np.shape(predictionsProb_)[0],np.shape(predictionsProb_)[1]))

        if len(np.shape(predictionsProb_))==2:
            predictionsProb_=np.reshape(predictionsProb_,(1,np.shape(predictionsProb_)[0],np.shape(predictionsProb_)[1]))
        
        
        nLabels=np.shape(y_target_)[1]
        
        if isinstance(classifierslabels,type(None)):
            classifierslabels=[f"L{i}" for i in range(nLabels)]
        allIMCP=pd.DataFrame()
        for il,label_ in enumerate(classifierslabels):
            y_target_curr=y_target_[:,il]
            IMCP=self.getIMCP(predictionsProb_[il],y_target_curr)
            IMCP['label']=label_

            #Random reference
            randClassifier_=self.getRamdomClassificationModel(Y_target=y_target_curr)
            IMCP_rand=self.getIMCP(randClassifier_[0],randClassifier_[1])
            IMCP_rand['label']=f"{label_}_random"
            
            allIMCP=pd.concat([allIMCP,IMCP,IMCP_rand],sort=False,ignore_index=True)

        return allIMCP
    

    def fixExtremePoints(self,imatrix,forCurve="ROC"):
        tmpDF=pd.DataFrame()
        if np.shape(imatrix)[1]!=3: return tmpDF

        if forCurve=="ROC":
            tmpDF=imatrix[ ~(((imatrix.iloc[:,0]<=1E-6) & (imatrix.iloc[:,1]<=1E-6)) |
            ((np.abs(imatrix.iloc[:,0]-1.0)<=1E-6) & (np.abs(imatrix.iloc[:,1]-1.0)<=1E-6)) |
            ((np.abs(imatrix.iloc[:,0]-1.0)<=1E-6) & (imatrix.iloc[:,1]<=1E-6)))].copy()
            tmpDF=tmpDF.sort_values(by="Prob",ascending=False).reset_index(drop=True)
            tmpDF.loc[len(tmpDF)]=[1.0,1.0,0.0]
            tmpDF.loc[-0.5]=[0.0,0.0,1.0]
            tmpDF=tmpDF.sort_index(axis=0).reset_index(drop=True)
        elif forCurve=="PRC":
            tmpDF=imatrix[ ~(((imatrix.iloc[:,0]<=1E-6) & (imatrix.iloc[:,1]<=1E-6)) |
            ((imatrix.iloc[:,0]<=1E-6) & (np.abs(imatrix.iloc[:,1]-1.0)<=1E-6)) |
            ((np.abs(imatrix.iloc[:,0]-1.0)<=1E-6) & (imatrix.iloc[:,1]<=1E-6)))].copy()
            tmpDF=tmpDF.sort_values(by="Prob",ascending=False).reset_index(drop=True)
            tmpDF.loc[len(tmpDF)]=[1.0,0.0,0.0]
            tmpDF.loc[-0.5]=[0.0,1.0,1.0]
            tmpDF=tmpDF.sort_index(axis=0).reset_index(drop=True)

        return  tmpDF    

    
    def getROC_and_PRC(self,metricForLabels,recallSpecificityColNames):

        probClassName=recallSpecificityColNames[0].replace("Recall","Prob")
        precClassName=recallSpecificityColNames[0].replace("Recall","Precision")
        metricsAtPthreshold=metricForLabels.copy()

        #ROC
        ROC_=metricForLabels[[recallSpecificityColNames[0]]].copy()
        ROC_['Fallout']=(100.0-metricsAtPthreshold[recallSpecificityColNames[1]])/100.0
        ROC_.columns=['Recall','Fallout']
        ROC_['Recall']=ROC_['Recall']/100.0
        ROC_=ROC_.iloc[:,[1,0]]
        ROC_['Prob']=metricsAtPthreshold[probClassName]
        ROC_=self.fixExtremePoints(ROC_)
        AUCROC=integrate.trapezoid(ROC_.Recall,ROC_.Fallout)
        ROC_['AUCROC']=AUCROC

        #PRC
        PRC_=metricForLabels[[recallSpecificityColNames[0],precClassName,probClassName]].copy()
        PRC_.columns=["Recall","Precision","Prob"]
        PRC_["Recall"]=PRC_["Recall"]/100.0
        PRC_["Precision"]=PRC_["Precision"]/100.0
        PRC_=self.fixExtremePoints(PRC_,forCurve="PRC")
        AUCPRC=integrate.trapezoid(PRC_.Precision,PRC_.Recall)
        PRC_['AUCPRC']=AUCPRC

        return (ROC_,PRC_)

    def getDiscreteProbability(self,y_target):
        y_target_=np.array(y_target).astype(int)
        classes=np.sort(np.unique(y_target_))
        offset=np.power(10,len(classes))
        clssDict={j:i for i,j in enumerate(classes)}
        return np.array([np.flip(list(str(offset+np.power(10,clssDict[lclss])))[1:],axis=0).astype(float) for lclss in y_target_])

    
    def getDeltaLabelsVector(self,y_target):
        y_target_=np.array(y_target).astype(int)
        classes=np.sort(np.unique(y_target_))
        numClasses=len(classes)
        clssDict={j:i for i,j in enumerate(classes)}
        classCoded=np.array([ clssDict[lclss] for lclss in y_target_])
        delta=[ 1.0/(numClasses*len(classCoded[classCoded==i])) for i in classCoded]
        return delta


    def getIMCP(self,predictionsProb,y_target):
        y_target_discrete=self.getDiscreteProbability(y_target)
        K=len(np.unique(y_target))
        Dv=np.sqrt(predictionsProb)-y_target_discrete
        phi=1.0-np.sqrt(np.sum(Dv * Dv, axis=1))/np.sqrt(2)
        d_=self.getDeltaLabelsVector(y_target)
        IMCP=pd.DataFrame({'phi':phi,'delta':d_,'instanceIdx':range(len(d_))}).sort_values(["phi"]).reset_index(drop=True)
        IMCP['x']=list(map(lambda i: np.sum(IMCP.delta[0:i])+IMCP.delta[i]/2.0,range(len(IMCP.delta))))
        AUCIMCP=integrate.trapezoid(IMCP.phi, IMCP.x)
        IMCP['AUCIMCP']=AUCIMCP
        lowerUBoundary=1.0-np.sqrt(1.0-1.0/np.sqrt(K))
        upperUBoundary=1.0-np.sqrt(1.0-np.sqrt(0.5))
        IMCP['1_minus_theta']=lowerUBoundary
        IMCP['x_lb']=IMCP.iloc[np.abs(IMCP.phi-lowerUBoundary).idxmin()].x
        IMCP['1_minus_eta']=upperUBoundary
        IMCP['x_ub']=IMCP.iloc[np.abs(IMCP.phi-upperUBoundary).idxmin()].x        
        return IMCP


    def getTestMetricsStatistics(self,inputDataSet,yColumnName,
                                 n_folds=10,stratificationColumn=None,nCategories=0,selectedProb=None,partitionFraction=0.5,
                                 inputModel=None,inputTransformation=None,inputPreprocessing=None,labelClass="-",mainAndComplementaryClasses=None,
                                 rand_seed=58):
        AUCs=pd.DataFrame()
        singleValsMetrics=pd.DataFrame()
        maxValsMetrics=pd.DataFrame()
        ovothAllSingleMetrics=pd.DataFrame()
        ROC_curves=pd.DataFrame()
        PRC_curves=pd.DataFrame()
        IMCP_curves=pd.DataFrame()
        metricsData=pd.DataFrame()
        inputPartition=pd.DataFrame()
        modelInputOutputs=pd.DataFrame()

        mainAndComplementaryClasses_=mainAndComplementaryClasses
        if isinstance(mainAndComplementaryClasses,list):
            mainAndComplementaryClasses_=mainAndComplementaryClasses
        elif isinstance(mainAndComplementaryClasses,str):
            if mainAndComplementaryClasses=="default":
                mainAndComplementaryClasses_=['Recall(1)','Recall(0)']

        np.random.seed(rand_seed)
        randSeeds=np.random.choice(range(0,10000), size=int(n_folds/2), replace=False)

        kfoldData=inputDataSet.copy()
        i=0

        if nCategories>0:
            if isinstance(stratificationColumn,type(None)):
                return 
            dataToStratify="stratification"
            kfoldData[dataToStratify]=pd.qcut(kfoldData[stratificationColumn],nCategories,labels=range(1,nCategories+1))
        else:
            if isinstance(stratificationColumn,type(None)):
                dataToStratify=yColumnName
            else:
                dataToStratify=stratificationColumn
        

        for l,seed_ in enumerate(randSeeds):
            kfold_A, kfold_B = train_test_split(
                kfoldData, 
                test_size=partitionFraction,
                stratify=kfoldData[dataToStratify],
                random_state=seed_,
                shuffle=True
            )

            for kfold_ in (kfold_A, kfold_B):
                i+=1

                y_kfold=kfold_[yColumnName].copy()
                X_kfold=kfold_.copy().drop(columns=[yColumnName])
                if not(isinstance(stratificationColumn,type(None))): X_kfold=X_kfold.drop(columns=[dataToStratify])

                X_FT_kfold=X_kfold
                if not(isinstance(inputPreprocessing,type(None))):
                    inputPreprocessing_=inputPreprocessing.transform(X_kfold)
                    if not(isinstance(inputTransformation,type(None))):
                        X_FT_kfold=inputTransformation.transform(inputPreprocessing_)
                    else:
                        X_FT_kfold=inputPreprocessing_
                        
                elif not(isinstance(inputTransformation,type(None))):
                    X_FT_kfold=inputTransformation.transform(X_kfold)
                    
                if isinstance(inputModel,type(None)):
                    inputPartition=pd.concat([inputPartition,pd.DataFrame({"X":[X_FT_kfold],"Y":[y_kfold],"kfold":[i]})],sort=False,ignore_index=True)
                else:            
                
                    y_test_kfold=y_kfold.to_numpy()
                    y_pred_prob_kfold=inputModel.predict_proba(X_FT_kfold)
                    modelInputOutputs=pd.concat([modelInputOutputs,pd.DataFrame({"X_FT":[X_FT_kfold],"Y_prediction":[y_pred_prob_kfold],"Y_target":[y_kfold],"kfold":[i]})],sort=False,ignore_index=True)

                    CFMTX=self.getSingleLabelContingenceMatrix(y_pred_prob_kfold,y_test_kfold)
                    maxProbMetric=self.getSingleLabelMetrics(CFMTX)
                    maxProbMetric['kfold']=[i]
                    maxValsMetrics=pd.concat([maxValsMetrics,maxProbMetric],sort=False,ignore_index=True)

                    IMCP_test_kfold=self.getIMCPCurves(y_pred_prob_kfold,y_test_kfold,classifierslabels=[labelClass])
                    IMCP_test_kfold['kfold']=i
                    IMCP_curves=pd.concat([IMCP_curves,IMCP_test_kfold],sort=False,ignore_index=True)


                    if not(isinstance(selectedProb,type(None))):
                        selectedProbMetric=self.getMetricsAtProbabilityThreshold(y_pred_prob_kfold,y_test_kfold,[labelClass],pThres=selectedProb)
                        selectedProbMetric['kfold']=[i]
                        singleValsMetrics=pd.concat([singleValsMetrics,selectedProbMetric],sort=False,ignore_index=True)


                    if isinstance(mainAndComplementaryClasses_,list):
                        testMetricAtPThreshold_kfold=self.getMetricsAtProbabilityThreshold(y_pred_prob_kfold,y_test_kfold,[labelClass],probSteps=50)
                        testMetricAtPThreshold_kfold['kfold']=i
                        metricsData=pd.concat([metricsData,testMetricAtPThreshold_kfold],sort=False,ignore_index=True)

                        ROC,PRC=self.getROC_and_PRC(testMetricAtPThreshold_kfold,[mainAndComplementaryClasses_[0],mainAndComplementaryClasses_[1]])
                        ROC['kfold']=i
                        PRC['kfold']=i
                        ROC_curves=pd.concat([ROC_curves,ROC],sort=False,ignore_index=True)
                        PRC_curves=pd.concat([PRC_curves,PRC],sort=False,ignore_index=True)

                        AUCs=pd.concat([AUCs,pd.DataFrame({"kfold":[i],
                                                           "AUC(IMCP)":[IMCP_test_kfold.AUCIMCP.iloc[0]],
                                                           "AUC(ROC)":[ROC.AUCROC.iloc[0]],
                                                           "AUC(PRC)":[PRC.AUCPRC.iloc[0]]
                                                          })],sort=False,ignore_index=True)
                    else:
                        AUCs=pd.concat([AUCs,pd.DataFrame({"kfold":[i],
                                                           "AUC(IMCP)":[IMCP_test_kfold.AUCIMCP.iloc[0]]
                                                           })],sort=False,ignore_index=True)

        if isinstance(inputModel,type(None)):
            return inputPartition

        return pd.DataFrame({"maxValsMetrics":[maxValsMetrics],"singleValsMetrics":[singleValsMetrics],"AUCs":[AUCs],"ROC_curves":[ROC_curves],"PRC_curves":[PRC_curves],"IMCP_curves":[IMCP_curves],"metricsData":[metricsData],"modelInputOutputs":[modelInputOutputs],"ovsOthMetrics":[ovothAllSingleMetrics]})
    
    
    def getMultiClassCrossValidationTrainingScores(self,inputModel,X,Y_target,classifierslabel="-",n_folds=10,partitionFraction=0.2,rand_seed=45,evalBinaryCurves=False):

        np.random.seed(rand_seed)
        random.seed(rand_seed)
        seeds_=random.sample(range(n_folds*100), k=n_folds)
        allSingleMetrics=pd.DataFrame()
        ovothAllSingleMetrics=pd.DataFrame()
        for seed_ in seeds_:
            X_kfold_train,X_kfold_validation,Y_target_kfold_train,Y_target_kfold_validation = train_test_split(
                X,
                Y_target,
                test_size=partitionFraction,
                stratify=Y_target,
                random_state=seed_,
                shuffle=True
            )

            rfmf=inputModel.fit(X_kfold_train,Y_target_kfold_train)
            Y_prediction_prob_kfold_validation=inputModel.predict_proba(X_kfold_validation)

            trainKfoldCFMTX=self.getSingleLabelContingenceMatrix(Y_prediction_prob_kfold_validation,Y_target_kfold_validation)
            sMetrics=self.getSingleLabelMetrics(trainKfoldCFMTX)

            if evalBinaryCurves:
                trainingKfoldPThresholdMetrics=self.getMetricsAtProbabilityThreshold(Y_prediction_prob_kfold_validation,
                                                                                         Y_target_kfold_validation,classifierslabel,
                                                                                         probSteps=50)
                
                ROC,PRC=self.getROC_and_PRC(trainingKfoldPThresholdMetrics,['Recall(1)','Recall(0)'])
                sMetrics["AUCROC"]=ROC.AUCROC.iloc[0]
                sMetrics["AUCPRC"]=PRC.AUCPRC.iloc[0]

            
            IMCP=self.getIMCP(Y_prediction_prob_kfold_validation,Y_target_kfold_validation)
            sMetrics["AUCIMCP"]=IMCP.AUCIMCP.iloc[0]
            allSingleMetrics=pd.concat([allSingleMetrics,sMetrics],sort=False,ignore_index=True)            
            
        data=allSingleMetrics.copy().T.to_numpy()
        allSingleMetrics=allSingleMetrics.agg(['mean','std',self.getUncertainty,'count']).reset_index(drop=False).T
        allSingleMetrics.columns=allSingleMetrics.loc['index']
        allSingleMetrics=allSingleMetrics.iloc[1:]
        allSingleMetrics.columns=[cname.replace("getUncertainty", "U").replace("count", "n") for cname in allSingleMetrics.columns] 
        allSingleMetrics=allSingleMetrics.reset_index(drop=False)
        allSingleMetrics=allSingleMetrics.rename(columns={"index":"score"})
        allSingleMetrics["data"]=list(data)
        allSingleMetrics["label"]=classifierslabel
        rfmf=inputModel.fit(X,Y_target)

            
        return allSingleMetrics,ovothAllSingleMetrics


    def kFoldsMultilabelSplit(self,idxs,labels,k_folds,test_size,rand_seed=None):
        np.random.seed(rand_seed)
        random.seed(rand_seed)
        
        multilabels=labels.copy().loc[idxs].reset_index(drop=True)
        if k_folds>1:
            seeds_=random.sample(range(k_folds*100), k=k_folds)
            for k_fold,int_rand_seed in enumerate(seeds_):
                np.random.seed(int_rand_seed)
                idxsRandomized=np.random.choice(range(0,len(idxs)), size=len(idxs), replace=False)
                X_trainSet,y_trainSet,X_testSet,y_testSet=iterative_train_test_split(np.reshape(idxsRandomized,(len(idxsRandomized),1)),multilabels.loc[idxsRandomized].to_numpy(), test_size = test_size)
                yield (list(X_trainSet[:,0]),list(X_testSet[:,0]))
        elif k_folds==1:
            idxsRandomized=np.random.choice(range(0,len(idxs)), size=len(idxs), replace=False)
            X_trainSet,y_trainSet,X_testSet,y_testSet=iterative_train_test_split(np.reshape(idxsRandomized,(len(idxsRandomized),1)),multilabels.loc[idxsRandomized].to_numpy(), test_size = test_size)
            return (list(X_trainSet[:,0]),list(X_testSet[:,0]))


    def getMultilabelCrossValidationTrainingScores(self,inputModel,X,Y_target,classifierslabels,multilabelSplit):

        allSingleMetrics=pd.DataFrame()
        for kfold in multilabelSplit:
            X_kfold_train=X[kfold[0]]
            Y_target_kfold_train=Y_target[kfold[0]]
            X_kfold_validation=X[kfold[1]]
            Y_target_kfold_validation=Y_target[kfold[1]]
            rfmf=inputModel.fit(X_kfold_train,Y_target_kfold_train)
            Y_prediction_prob_kfold_validation=inputModel.predict_proba(X_kfold_validation)
            nLabels=np.shape(Y_prediction_prob_kfold_validation)[0]
            for i in range(nLabels):
                trainKfoldCFMTX=self.getSingleLabelContingenceMatrix(Y_prediction_prob_kfold_validation[i],Y_target_kfold_validation[:,i])
                sMetrics=self.getSingleLabelMetrics(trainKfoldCFMTX)

                #Curves
                trainingKfoldPThresholdMetrics=self.getMetricsAtProbabilityThreshold(Y_prediction_prob_kfold_validation[i],
                                                                                         Y_target_kfold_validation[:,i],
                                                                                         [classifierslabels[i]],probSteps=50)
                IMCP=self.getIMCP(Y_prediction_prob_kfold_validation[i],Y_target_kfold_validation[:,i])
                sMetrics["AUCIMCP"]=IMCP.AUCIMCP.iloc[0]
                ROC,PRC=self.getROC_and_PRC(trainingKfoldPThresholdMetrics,['Recall(1)','Recall(0)'])
                sMetrics["AUCROC"]=ROC.AUCROC.iloc[0]
                sMetrics["AUCPRC"]=PRC.AUCPRC.iloc[0]



                sMetrics["label"]=classifierslabels[i]
                allSingleMetrics=pd.concat([allSingleMetrics,sMetrics],sort=False,ignore_index=True)

        summaryMetrics=pd.DataFrame()
        for label_ in classifierslabels:
            data=allSingleMetrics[allSingleMetrics.label==label_].drop(columns=["label"]).T.to_numpy()
            currLabel=allSingleMetrics[allSingleMetrics.label==label_]
            currLabel=currLabel.drop(columns=["label"])
            currLabel=currLabel.agg(['mean','std',self.getUncertainty,'count']).reset_index(drop=False)
            currLabel=currLabel.T
            currLabel.columns=currLabel.loc['index']
            currLabel=currLabel.iloc[1:]
            currLabel.columns=[cname.replace("getUncertainty", "U").replace("count", "n") for cname in currLabel.columns] 
            currLabel=currLabel.reset_index(drop=False)
            currLabel=currLabel.rename(columns={"index":"score"})
            currLabel["data"]=list(data)
            currLabel["label"]=[label_]*len(currLabel)
            summaryMetrics=pd.concat([summaryMetrics,currLabel],sort=False,ignore_index=True)

        rfmf=inputModel.fit(X,Y_target)
        return summaryMetrics        


    def getMultiLabelContingenceMatrix(self,predProb,y_target,classNames=None,pThres=None):
        nLabels=np.shape(predProb)[0]
        cfxMtxs=()
        for i in range(nLabels):
            scm=self.getSingleLabelContingenceMatrix(predProb[i],y_target[:,i],classNames=classNames,pThres=pThres)
            cfxMtxs=cfxMtxs+(scm,)
        return cfxMtxs
    
    def getMultilabelTestMetricsStatistics(self,inputDataSet,Y_targets,
                                 n_folds=10,selectedProb=None,partitionFraction=0.5,
                                 inputModel=None,inputTransformation=None,inputPreprocessing=None,mainAndComplementaryClasses=None,
                                 rand_seed=58):
        AUCs=pd.DataFrame()
        singleValsMetrics=pd.DataFrame()
        maxValsMetrics=pd.DataFrame()
        ROC_curves=pd.DataFrame()
        PRC_curves=pd.DataFrame()
        IMCP_curves=pd.DataFrame()
        metricsData=pd.DataFrame()
        inputPartition=pd.DataFrame()
        modelInputOutputs=pd.DataFrame()

        mainAndComplementaryClasses_=mainAndComplementaryClasses
        if isinstance(mainAndComplementaryClasses,list):
            mainAndComplementaryClasses_=mainAndComplementaryClasses
        elif isinstance(mainAndComplementaryClasses,str):
            if mainAndComplementaryClasses=="default":
                mainAndComplementaryClasses_=['Recall(1)','Recall(0)']

        if not(isinstance(selectedProb,type(None))):
            if isinstance(selectedProb,int):
                selectedProb_=[selectedProb]*np.shape(Y_targets)[1]
            else:
                selectedProb_=selectedProb
                
        np.random.seed(rand_seed)
        kfoldData=inputDataSet.copy()
        i=0
        
        kfolds=self.kFoldsMultilabelSplit(Y_targets.index,Y_targets,k_folds=int(n_folds/2),test_size=partitionFraction,rand_seed=rand_seed)
        l=-1
        
        for kfold_ in kfolds:
            l+=1

            for kfold_ in (kfoldData.loc[kfold_[0]], kfoldData.loc[kfold_[1]]):
                i+=1

                # Setting input and output data
                y_kfold=Y_targets.loc[kfold_.index].copy()
                X_kfold=kfold_.copy()

                X_FT_kfold=X_kfold
                if not(isinstance(inputPreprocessing,type(None))):
                    inputPreprocessing_=inputPreprocessing.transform(X_kfold)
                    if not(isinstance(inputTransformation,type(None))):
                        X_FT_kfold=inputTransformation.transform(inputPreprocessing_)
                    else:
                        X_FT_kfold=inputPreprocessing_
                        
                elif not(isinstance(inputTransformation,type(None))):
                    X_FT_kfold=inputTransformation.transform(X_kfold)
                    
                if isinstance(inputModel,type(None)):
                    inputPartition=pd.concat([inputPartition,pd.DataFrame({"X":[X_FT_kfold],"Y":[y_kfold],"kfold":[i]})],sort=False,ignore_index=True)
                else:            

                    y_preds_prob_kfold=inputModel.predict_proba(X_FT_kfold)
                    modelInputOutputs=pd.concat([modelInputOutputs,pd.DataFrame({"X_FT":[X_FT_kfold],"Y_predictions":[y_preds_prob_kfold],"Y_target":[y_kfold],"kfold":[i]})],sort=False,ignore_index=True)
                    for il,label_ in enumerate(Y_targets):

                        y_test_kfold=y_kfold.iloc[:,il].to_numpy()
                        y_pred_prob_kfold=y_preds_prob_kfold[il]


                        CFMTX=self.getSingleLabelContingenceMatrix(y_pred_prob_kfold,y_test_kfold)
                        maxProbMetric=self.getSingleLabelMetrics(CFMTX)
                        maxProbMetric['label']=[label_]
                        maxProbMetric['kfold']=[i]
                        maxValsMetrics=pd.concat([maxValsMetrics,maxProbMetric],sort=False,ignore_index=True)

                        IMCP_test_kfold=self.getIMCPCurves(y_pred_prob_kfold,y_test_kfold,classifierslabels=[label_])
                        IMCP_test_kfold['kfold']=i
                        IMCP_curves=pd.concat([IMCP_curves,IMCP_test_kfold],sort=False,ignore_index=True)


                        if not(isinstance(selectedProb,type(None))):
                            selectedProbMetric=self.getMetricsAtProbabilityThreshold(y_pred_prob_kfold,y_test_kfold,[label_],pThres=selectedProb_[il])
                            selectedProbMetric['label']=[label_]
                            selectedProbMetric['kfold']=[i]
                            singleValsMetrics=pd.concat([singleValsMetrics,selectedProbMetric],sort=False,ignore_index=True)


                        if isinstance(mainAndComplementaryClasses_,list):
                            testMetricAtPThreshold_kfold=self.getMetricsAtProbabilityThreshold(y_pred_prob_kfold,y_test_kfold,[label_],probSteps=50)
                            testMetricAtPThreshold_kfold['label']=label_
                            testMetricAtPThreshold_kfold['kfold']=i
                            metricsData=pd.concat([metricsData,testMetricAtPThreshold_kfold],sort=False,ignore_index=True)

                            ROC,PRC=self.getROC_and_PRC(testMetricAtPThreshold_kfold,[mainAndComplementaryClasses_[0],mainAndComplementaryClasses_[1]])
                            ROC['label']=label_
                            ROC['kfold']=i
                            PRC['label']=label_
                            PRC['kfold']=i
                            ROC_curves=pd.concat([ROC_curves,ROC],sort=False,ignore_index=True)
                            PRC_curves=pd.concat([PRC_curves,PRC],sort=False,ignore_index=True)

                            AUCs=pd.concat([AUCs,pd.DataFrame({"kfold":[i],
                                                               "AUC(IMCP)":[IMCP_test_kfold.AUCIMCP.iloc[0]],
                                                               "AUC(ROC)":[ROC.AUCROC.iloc[0]],
                                                               "AUC(PRC)":[PRC.AUCPRC.iloc[0]],
                                                               "label":[label_]
                                                              })],sort=False,ignore_index=True)
                        else:
                            AUCs=pd.concat([AUCs,pd.DataFrame({"kfold":[i],
                                                               "AUC(IMCP)":[IMCP_test_kfold.AUCIMCP.iloc[0]],
                                                               "label":[label_]
                                                               })],sort=False,ignore_index=True)


        if isinstance(inputModel,type(None)):
            return inputPartition

        return pd.DataFrame({"maxValsMetrics":[maxValsMetrics],"singleValsMetrics":[singleValsMetrics],"AUCs":[AUCs],"ROC_curves":[ROC_curves],"PRC_curves":[PRC_curves],"IMCP_curves":[IMCP_curves],"metricsData":[metricsData],"modelInputOutputs":[modelInputOutputs]})


    def getConfusionMatrixOneVsOthers(self,predProb,y_target,selectedClass,classNames=None,pThres=None):
        y_target_=y_target.astype(int)
        if isinstance(classNames,type(None)):
            self.__classesNames=np.sort(np.unique(y_target.astype(int))).astype(str)
        else:
            self.__classesNames=np.array(classNames).astype(str)
        self.__numberOfClasses=len(self.__classesNames)
        if self.__numberOfClasses<=2: return pd.DataFrame()

        if isinstance(pThres,type(None)):
            self.__pTh=['max']*self.__numberOfClasses
            predictionsLabs=np.array([list(instance>=max(instance)) for instance in predProb])
        else:
            self.__pTh=pThres
            if not(isinstance(pThres,list)):
                self.__pTh=[pThres]*len(self.__classesNames)
            predictionsLabs=np.array(list(map(lambda i:list(predProb[:,i]>=(self.__pTh[i])),range(np.shape(predProb)[1])))).transpose()

        predictionsLabs=predictionsLabs.astype(str)
        predictionsLabs=np.char.replace(predictionsLabs, 'False', '')
        y_target_=y_target_.astype(str)

        notSelectedClasess="-".join(self.__classesNames).replace("-"+selectedClass,"").replace(selectedClass+"-","")
        for i,class_ in enumerate(self.__classesNames):

            if class_!=selectedClass: 
                predictionsLabs[:,i]=np.char.replace(predictionsLabs[:,i],'True',"J_v")
                y_target_=np.char.replace(y_target_,class_,"J_v")
            else:
                predictionsLabs[:,i]=np.char.replace(predictionsLabs[:,i],'True',class_)
        predictionsLabs=[''.join(row) for row in predictionsLabs]

        cfMtx=pd.DataFrame()
        __newClasses=["J_v",selectedClass]
        __newClasses.sort()

        for idx,class_ in enumerate(__newClasses):
            clabels=np.array(np.array(y_target_).astype(str))==class_
            classPredictions=pd.DataFrame(np.unique(np.array(predictionsLabs)[clabels],return_counts=True))
            classPredictions.columns=classPredictions.iloc[0]
            classPredictions=classPredictions.drop(0,axis=0).reset_index(drop=True)
            overlapedClasses=list(set(classPredictions.columns)-set(__newClasses))

            emptyClasses=list(set(__newClasses)-set(classPredictions.columns))
            emptyClasses = classPredictions.columns.tolist() + emptyClasses
            classPredictions = classPredictions.reindex(columns=emptyClasses)
            classPredictions=classPredictions.replace(np.nan, 0)
            for oClass in overlapedClasses:
                if class_ in oClass: classPredictions.loc[0,class_]=classPredictions[class_].iloc[0]+classPredictions.loc[0,oClass]
            classPredictions=classPredictions[__newClasses]


            classPredictions['target']=class_
            cfMtx=pd.concat([cfMtx,classPredictions])    

        cfMtx=cfMtx.sort_index(axis=1).set_index('target')
        cfMtx.columns.name='prediction'
        cfMtx.columns=np.char.replace(cfMtx.columns.to_list(),'J_v',notSelectedClasess)
        cfMtx=cfMtx.transpose()
        cfMtx.columns=np.char.replace(cfMtx.columns.to_list(),'J_v',notSelectedClasess)
        cfMtx.columns.name='target'
        cfMtx.index.name='prediction'
        self.__cfMtx=cfMtx
        return cfMtx    


    
    def getIMCPAveragedCurves(self,IMCPavg,statData):
    
        statData_=statData.iloc[0].IMCP_curves
        labels_=np.unique(statData_.label)[[ not("random" in x) for x in np.unique(statData_.label)]]     
        allAvgCurves=pd.DataFrame()
        for label_ in labels_:
            IMCPavg_=IMCPavg[IMCPavg.label==label_].copy()
            IMCPavg_=IMCPavg_.groupby("phi").x.agg(['min'])
            curr_IMCP_curves=statData_[statData_.label==label_].copy().sort_values("phi",ascending=True)
            curr_IMCP_curves['n']=curr_IMCP_curves.groupby(['phi']).phi.transform('count')
            curr_IMCP_curves=curr_IMCP_curves.groupby(['phi'])[['phi','x','n']].agg(['mean','std',self.getUncertainty])
            curr_IMCP_curves=curr_IMCP_curves.T.reset_index(drop=False).T
            cols=list(map(lambda x:x[0]+"_"+x[1] ,zip(curr_IMCP_curves.loc["level_0"],curr_IMCP_curves.loc["level_1"])))
            curr_IMCP_curves.columns=cols
            curr_IMCP_curves=curr_IMCP_curves.iloc[2:,:].reset_index(drop=True)
            curr_IMCP_curves=curr_IMCP_curves.set_index("phi_mean",drop=False).join(IMCPavg_)
            curr_IMCP_curves['label']=label_
            curr_IMCP_curves.columns=[cname.replace("getUncertainty", "U").replace("n_mean", "n") for cname in curr_IMCP_curves.columns]
            curr_IMCP_curves=curr_IMCP_curves.rename(columns={"min":"x_avg"})
            curr_IMCP_curves=curr_IMCP_curves.drop(columns=["n_std","n_U"])
            allAvgCurves=pd.concat([allAvgCurves,curr_IMCP_curves],sort=False,ignore_index=True)
            
        return allAvgCurves


    def summarizeResults(self,RT_results,dtR=0.05,lpoints=3000,upoints=5000,estimator="mean"):
        nbins=round((RT_results.tR_pred.max()-RT_results.tR_pred.min())/dtR)
        bins=np.histogram(RT_results.tR_pred,bins=nbins)
        bins=pd.DataFrame(bins).transpose()
        lowerSelectectedBins=bins[0]>lpoints
        lowerSelectectedBins.iloc[0]=True
        lbins=bins[lowerSelectectedBins][1].to_list()
        upperSelectectedBins=bins[0]>upoints
        ubins=np.array(bins[upperSelectectedBins][1].to_list())
        bins=np.append(lbins,ubins[np.array(ubins)>lbins[-1]])
        RT_results['binGroup']=np.digitize(RT_results.tR_pred,bins)
        RT_results=RT_results.sort_values(['tR_pred']).reset_index(drop=True)
        lnbin=RT_results['binGroup'].max()-1
        lidx=RT_results.index.max()
        RT_results.at[lidx,'binGroup']=lnbin
        RMSE_model=np.sqrt(  np.sum((RT_results.tR_pred-RT_results.tR_exp)**2)/ len(RT_results))
        RMSErrByBin=RT_results.groupby('binGroup').apply(lambda x:  np.sqrt(np.sum((x['tR_pred']-x['tR_exp'])**2)/len(x)),include_groups=False )
        nData=RT_results.groupby('binGroup').nAssess.apply("count")
        RT_results=RT_results.groupby('binGroup')[["tR_pred","tR_exp"]].agg(['mean','std'])
        RT_results=RT_results.T.reset_index(drop=False).T
        cols=list(map(lambda x:x[0]+"_"+x[1] ,zip(RT_results.loc["level_0"],RT_results.loc["level_1"])))
        RT_results.columns=cols
        RT_results=RT_results.iloc[2:,:].reset_index(drop=False)
        RT_results["n"]=nData.reset_index(drop=True)
        RT_results["RMSError"]=RMSErrByBin.reset_index(drop=True)
        RT_results["U.95"]=RT_results.groupby('binGroup').apply(lambda x: (stats.t.interval(0.95,x['n']-1)[1])[0]*x['RMSError'],include_groups=False).to_list()
        RT_results['tR_lb']=RT_results.tR_pred_mean-RT_results['U.95']
        RT_results['tR_ub']=RT_results.tR_pred_mean+RT_results['U.95']
        RT_results["RMSE_model"]=RMSE_model

        return RT_results.sort_values("tR_exp_mean").reset_index(drop=True).drop(columns=['binGroup'])



    def getRTModelConfidenceInterval(self,uFile,X_data,Y_data,data_idxs,model,numOfTests=1500):
        total_assessments=int(numOfTests**2/50)
        if not( (os.path.exists(uFile) | os.path.exists(uFile.replace(".tsv",".tar.gz")))):
            compiledResults=pd.DataFrame()
            pbarSpects = tqdm(total=total_assessments, bar_format='{l_bar}{bar:100}{r_bar}{bar:-5b}')

            nAssess=0
            random.seed(42)
            for ranState in random.sample(range(int(numOfTests/5)),k=int(numOfTests/50)):
                for set_ in self.getTrainingTestSetsRandom(data_idxs,int(numOfTests),randomState=ranState):

                    startTime=time.monotonic()
                    pbarSpects.set_description(f"Processing: {nAssess+1}/{total_assessments} ")    
                    clear_output(wait=True)

                    trainingSet_,testSet_=set_
                    X_train_U=X_data[trainingSet_,]
                    Y_train_U=Y_data[trainingSet_]                
                    X_test_U=X_data[testSet_,]
                    Y_test_U=Y_data[testSet_]

                    model.fit(X_train_U,Y_train_U)
                    tR_predictions_test=model.predict(X_test_U)
                    compiledResults=self.assessmentResults(Y_test_U,tR_predictions_test,nAssess)
                    if nAssess==0:
                        compiledResults.to_csv(uFile,sep="\t",index=False)
                    else:
                        compiledResults.to_csv(uFile,sep="\t",index=False, mode='a',header=False)

                    pbarSpects.update(1)
                    clear_output(wait=True)
                    nAssess+=1
            pbarSpects.bar_format       
            pbarSpects.close()  

        uFile_=uFile
        if not(os.path.exists(uFile)): uFile_=uFile.replace(".tsv",".tar.gz")
        if not(os.path.exists(uFile_)): return

        RT_summary=pd.read_csv(uFile_,sep="\t")
        RT_summary=self.summarizeResults(RT_summary)
        return RT_summary     
