import sys,os,glob
from pathlib import Path
import time
from datetime import  datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from decimal import Decimal
from scipy.stats import f
from scipy import integrate

from IPython.display import display, clear_output
from tqdm import tqdm
import warnings

class msIdentificands():
    
    def __init__(self):
                
        #Data imputation
        self.__defaultPImputation=0.5

        #Analysis framework
        self.__identificandsData=pd.DataFrame()
        self.__identificandsComponents=pd.DataFrame()
        self.__componentsToExclude=['l_noise','l_chromProblems']
        self.__blankPath=""
        self.zeroThreshold=1E-12
        self.__dIntThres=0.2
        
        ##Adquisition info
        self.__rawFile=""
        self.__rawFileBaseName=""
        self.__targetDetectionDataFile=""
        self.__targetSuspectsDetectionDataFile=""
        self.__targetKnownsDetectionDataFile=""
        self.__nontargetedDetectionDataFile=""
        self.__resultsPath="./resultsData"
        self.__bkresultsPath="./resultsData"
        self.__tR_uncertainty=0.1
        self.__weightingFactors=[]


    @property
    def existTargetDetectionDataFile(self):
        return os.path.exists(self.__targetDetectionDataFile)

    @property
    def existTargetSuspectsDetectionDataFile(self):
        if not(os.path.exists(self.__targetDetectionDataFile)):
            return os.path.exists(self.__targetSuspectsDetectionDataFile)
        else:
           return  "suspect-screening" in pd.read_csv(self.__targetDetectionDataFile,sep="\t")["identificationApproach"].unique()

    @property
    def existTargetKnownsDetectionDataFile(self):
        if not(os.path.exists(self.__targetDetectionDataFile)):
            return os.path.exists(self.__targetKnownsDetectionDataFile)
        else:
            return  "targeted-analysis" in pd.read_csv(self.__targetDetectionDataFile,sep="\t")["identificationApproach"].unique()


    @property
    def existNontargetedDetectionDataFile(self):
        return os.path.exists(self.__nontargetedDetectionDataFile)

        
    @property
    def rawFile(self):
        return self.__rawFile


    @rawFile.setter
    def rawFile(self,file):
        if os.path.exists(file):
            self.__rawFile=str(Path(file).resolve())
            self.__rawFileBaseName=os.path.basename(self.__rawFile).replace('.raw','')
            self.__resultsPath=os.path.join(self.__bkresultsPath,self.__rawFileBaseName)
            self.setAssesmentData()

    @property
    def rawFileBaseName(self):
        return self.__rawFileBaseName
            
    @rawFileBaseName.setter
    def rawFileBaseName(self,value):
        self.__rawFileBaseName=value

    @property
    def blankDIntThreshold(self):
        return self.__dIntThres
            
    @blankDIntThreshold.setter
    def blankDIntThreshold(self,value):
        self.__dIntThres=value

        
    @property
    def blankPath(self):
        return self.__blankPath

    @blankPath.setter
    def blankPath(self,value):
        self.__blankPath=value

    @property
    def resultsPath(self):
        return self.__resultsPath

    
    @resultsPath.setter
    def resultsPath(self,value):
        self.__bkresultsPath=value
        if not(os.path.exists(self.__bkresultsPath)):
            os.makedirs(self.__bkresultsPath)
        self.__resultsPath=os.path.join(self.__bkresultsPath,self.__rawFileBaseName)
                      

    @property
    def defaultProbabilityImputation(self):
        return self.__defaultPImputation

    @defaultProbabilityImputation.setter
    def defaultProbabilityImputation(self,value):
        self.__defaultPImputation=value

    @property
    def tR_uncertainty(self):
        return self.__tR_uncertainty

    @tR_uncertainty.setter
    def tR_uncertainty(self,value):
        self.__tR_uncertainty=value
        
    
    @property
    def componentsToExclude(self):
        return self.__componentsToExclude

    @componentsToExclude.setter
    def componentsToExclude(self,value):
        self.__componentsToExclude=value
          
    @property
    def identificandsData(self):
        return self.__identificandsData


    @identificandsData.setter
    def identificandsData(self,value):
        self.__identificandsData=value


    @property
    def identificandsComponents(self):        
        if not(self.__identificandsComponents.empty):
            return self.__identificandsData[['CompNum','canonicalSmiles','name','identificationApproach']].join(self.__identificandsComponents)
     

    @identificandsComponents.setter
    def identificandsComponents(self,value):
        self.__identificandsComponents=value

        
            
    def setAssesmentData(self):

        mMDataFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsData.tsv")
                
        self.__targetDetectionDataFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_targetedScreeningAllDetectedCompounds.tsv")
        self.__targetKnownsDetectionDataFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_targetedAnalysisAllDetectedCompounds.tsv")
        self.__targetSuspectsDetectionDataFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_suspectScreeningAllDetectedCompounds.tsv")
        self.__nontargetedDetectionDataFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningAllDetectedCompounds.tsv")
        self.__identificandsData=pd.DataFrame()
        self.__identificandsComponents=pd.DataFrame()
        __tdd=None

        if os.path.exists(self.__nontargetedDetectionDataFile):
            try:
                self.__identificandsData=pd.read_csv(self.__nontargetedDetectionDataFile,sep="\t")
            except:
                None


        if os.path.exists(self.__targetDetectionDataFile):
            __tdd=None
            __compNumOffset=0
            if not(self.__identificandsData.empty): __compNumOffset=self.__identificandsData.CompNum.max()

            try:
                __tdd=pd.read_csv(self.__targetDetectionDataFile,sep="\t")
                __tdd.CompNum=__tdd.CompNum+__compNumOffset
                self.__identificandsData=pd.concat([self.__identificandsData,__tdd],sort=False,ignore_index=True)
            except:
                None


        if os.path.exists(self.__targetSuspectsDetectionDataFile):
            __tdd=None
            __compNumOffset=0
            if not(self.__identificandsData.empty): __compNumOffset=self.__identificandsData.CompNum.max()

            try:
                __tdd=pd.read_csv(self.__targetSuspectsDetectionDataFile,sep="\t")
                __tdd.CompNum=__tdd.CompNum+__compNumOffset
                self.__identificandsData=pd.concat([self.__identificandsData,__tdd],sort=False,ignore_index=True)
            except:
                None


        if os.path.exists(self.__targetKnownsDetectionDataFile):
            __tdd=None
            __compNumOffset=0
            if not(self.__identificandsData.empty):__compNumOffset=self.__identificandsData.CompNum.max()

            try:
                __tdd=pd.read_csv(self.__targetKnownsDetectionDataFile,sep="\t")           
                __tdd.CompNum=__tdd.CompNum+__compNumOffset
                self.__identificandsData=pd.concat([self.__identificandsData,__tdd],sort=False,ignore_index=True)
            except:
                None
               
        if not(self.__identificandsData.empty):
            self.__identificandsData.columns=";".join(self.__identificandsData.columns.to_list()).replace(";p_",";l_").split(";")
            self.__identificandsData=self.__identificandsData.rename(columns={'l_acceptableSignal':'l_chromSignalQlty',
                                                                          'l_vectorialProj':'l_isotopicProfile',
                                                                          'l_ms2':'l_ms2Annotation',
                                                                          'iMax2tRmax':'intensity',
                                                                          'tRmax':'tR'
                                                                          })                        
            self.__identificandsComponents=self.__identificandsData.T[np.bool_(self.__identificandsData.columns.str.find("l_").to_numpy()+1)].T.copy().drop(columns=self.__componentsToExclude)
            self.__identificandsData.to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsData.tsv"),sep="\t",index=False)
            self.__identificandsData[['CompNum','canonicalSmiles','name','identificationApproach','intensity','tR']].join(self.__identificandsComponents).to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsComponents.tsv"),sep="\t",index=False)

            self.getIdentificandsNaiveProbabilityCombination(saveResults=True)
            self.getIdentificandsBestNaiveProbability(saveResults=True)
            self.getSummaryInfo()
        return
                    
    def p_Lp(self,q,lp=2.0):
        return 1.0-(np.sum((1.0-q)**lp,axis=1)/(np.shape(q)[1]))**(1.0/lp)

    def p_hellinger(self,q):
        return 1.0-np.sqrt(np.sum((1.0-np.sqrt(q.astype(float)))**2,axis=1))/np.sqrt(np.shape(q)[1])

    def p_area(self,q):
        return np.sum(q/np.shape(q)[1],axis=1)

    def p_area_trapezoid(self,q):
        q_ord=np.sort(q, axis=1)[:, ::-1]
        x=np.tile(np.array(range(np.shape(q)[1]))/(np.shape(q)[1]-1.0), (np.shape(q)[0], 1)) 
        return integrate.trapezoid(q_ord,x)

    def p_likehood(self,q):
        return np.prod(q,axis=1)

            
    def getIdentificandsNaiveProbabilityCombination(self,compNum=None,excludeComps=None,saveResults=True,zeroThreshold=1E-6,zeroReplacement=1E-6):

        if self.__identificandsComponents.empty: return pd.DataFrame()

        __naiveCombinationComps=self.__identificandsComponents.copy()
        __naiveCombinationComps.l_ms2Qlty=__naiveCombinationComps.l_ms2Qlty.astype(float).fillna(self.defaultProbabilityImputation)
        __naiveCombinationComps.l_ms2Annotation=__naiveCombinationComps.l_ms2Annotation.astype(float).fillna(self.defaultProbabilityImputation)
        __naiveCombinationComps['l_ms2']=__naiveCombinationComps.l_ms2Qlty*__naiveCombinationComps.l_ms2Annotation
        __naiveCombinationComps=__naiveCombinationComps.drop(columns=['l_ms2Annotation','l_ms2Qlty'])

        if isinstance(excludeComps,list):__naiveCombinationComps=__naiveCombinationComps.drop(columns=excludeComps)

        ## Weight the retention time
        l_tR_idx=list(__naiveCombinationComps.columns).index("l_tR_inference")
        w_tR=1.0/(1.0+np.log10(np.maximum(self.__identificandsData.U_tR_inference/self.tR_uncertainty,1.0)+1E-16))
        __naiveCombinationComps.iloc[:,l_tR_idx]=__naiveCombinationComps.iloc[:,l_tR_idx]*w_tR
        p_comps=__naiveCombinationComps.to_numpy()

        __naiveCombinationComps.insert(0,'massError_ppm',self.__identificandsData.massError_ppm) 
        __naiveCombinationComps.insert(0,'w_tR',w_tR) 
        __naiveCombinationComps.insert(0,'U_tR_inference',self.__identificandsData.U_tR_inference) 
        __naiveCombinationComps.insert(0,'tR_inference',self.__identificandsData.tR_inference) 
        __naiveCombinationComps.insert(0,'p_area',self.p_area_trapezoid(p_comps)) 
        __naiveCombinationComps.insert(0,'p_hellinger',self.p_hellinger(p_comps)) 
        __naiveCombinationComps.insert(0,'p_Lp_1.5',self.p_Lp(p_comps,1.5)) 
        __naiveCombinationComps.insert(0,'p_Lp_2.5',self.p_Lp(p_comps,2.5)) 
        __naiveCombinationComps.insert(0,'p_euclidean',self.p_Lp(p_comps,2.0))        
        __naiveCombinationComps.insert(0,'l_identification',self.p_likehood(p_comps))
        __naiveCombinationComps['Imputed']=self.__identificandsComponents.isna().T.apply('any')
        __naiveCombinationComps=self.__identificandsData[['CompNum','canonicalSmiles','inchikey','name','bmonoisotopicMass','identificationApproach','intensity','tR','numMS2Fragments']].join(__naiveCombinationComps)

        if saveResults:
            tmpComps=__naiveCombinationComps.copy().infer_objects(copy=False).replace(0.0,zeroReplacement)
            tmpComps.join(self.getEffectsAndContributions(tmpComps)).to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsProbabilityWithoutFiltering.tsv"),sep="\t",index=False)

            
        __naiveCombinationComps=__naiveCombinationComps.drop(columns=["numMS2Fragments"])
        __naiveCombinationComps=__naiveCombinationComps[__naiveCombinationComps.l_identification>=zeroThreshold]
        __naiveCombinationCompsTmp=__naiveCombinationComps.copy()
        __naiveCombinationCompsTmp=__naiveCombinationCompsTmp[__naiveCombinationCompsTmp.identificationApproach=="non-targeted-screening"]
        
        tmpComps=__naiveCombinationComps.copy()
        intGrpIdxs=tmpComps[tmpComps.identificationApproach=="suspect-screening"].reset_index(drop=False)
        if not(intGrpIdxs.empty):
            intGrpIdxs=intGrpIdxs.groupby('intensity',as_index=False)[['index','l_identification']].apply("max")['index'].to_list()
            tmpComps=tmpComps.loc[intGrpIdxs].sort_values('CompNum').reset_index(drop=True)
            __naiveCombinationCompsTmp=pd.concat([__naiveCombinationCompsTmp,tmpComps],sort=False,ignore_index=True)


        tmpComps=__naiveCombinationComps.copy()
        intGrpIdxs=tmpComps[tmpComps.identificationApproach=="targeted-analysis"].reset_index(drop=False)
        if not(intGrpIdxs.empty):
            intGrpIdxs=intGrpIdxs.groupby('intensity',as_index=False)[['index','l_identification']].apply("max")['index'].to_list()
            tmpComps=tmpComps.loc[intGrpIdxs].sort_values('CompNum').reset_index(drop=True)      
            __naiveCombinationCompsTmp=pd.concat([__naiveCombinationCompsTmp,tmpComps],sort=False,ignore_index=True)
            
        __naiveCombinationComps=__naiveCombinationCompsTmp.copy()
        del __naiveCombinationCompsTmp
        del intGrpIdxs

        
        __naiveCombinationComps.insert(8,'p_candidate',__naiveCombinationComps.groupby('CompNum')['l_identification'].apply(lambda x:x/sum(x)).to_list())
        __naiveCombinationComps=__naiveCombinationComps.set_index('CompNum',drop=False).join(pd.DataFrame(__naiveCombinationComps.groupby('CompNum',group_keys=False).size())).reset_index(drop=True)
        
        __naiveCombinationComps=__naiveCombinationComps.set_index('CompNum',drop=False).join(__naiveCombinationComps.groupby(['CompNum'])[['l_identification']].apply('max').rename(columns={'l_identification':'lGrp'})).reset_index(drop=True)
        __naiveCombinationComps.insert(9,'candidatesNum',__naiveCombinationComps[0])
        __naiveCombinationComps=__naiveCombinationComps.drop([0],axis=1)

        
        __naiveCombinationComps['detectionCompNum']=__naiveCombinationComps['CompNum']
        __ordIdxs=__naiveCombinationComps['lGrp'].sort_values(ascending=False).index.to_list()
        __naiveCombinationComps=__naiveCombinationComps.iloc[__ordIdxs]
        __numer=__naiveCombinationComps[['CompNum']].copy().drop_duplicates()
        __numer['NumTmp']=range(1,len(__numer)+1)
        __naiveCombinationComps=__naiveCombinationComps.set_index('CompNum').join(__numer.set_index('CompNum')).set_index('NumTmp').reset_index()
        __naiveCombinationComps=__naiveCombinationComps.drop(columns=['lGrp']).rename(columns={'NumTmp':'CompNum'}).sort_values(['CompNum'],ignore_index=True)
        __naiveCombinationComps['p_candidate']=["-" if Ncomps==1 else __naiveCombinationComps.p_candidate[i] for i,Ncomps in enumerate(__naiveCombinationComps.candidatesNum) ]
        __naiveCombinationComps['candidatesNum']=["-" if ( (Ncomps==1) & (__naiveCombinationComps.identificationApproach[i]!="non-targeted-screening") ) else Ncomps for i,Ncomps in enumerate(__naiveCombinationComps.candidatesNum) ]
        
        
        if isinstance(compNum,int):
            __compNum=max(compNum%(__naiveCombinationComps.CompNum.max()+1),1)
            __naiveCombinationComps=__naiveCombinationComps[__naiveCombinationComps.CompNum==__compNum]


        __naiveCombinationComps=__naiveCombinationComps.join(self.getEffectsAndContributions(__naiveCombinationComps))
        if saveResults: __naiveCombinationComps.to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsProbability.tsv"),sep="\t",index=False)        
        return __naiveCombinationComps

    
    def getIdentificandsBestNaiveProbability(self,compNum=None,saveResults=True):

        if self.__identificandsComponents.empty: return
        __bestProb=self.getIdentificandsNaiveProbabilityCombination().copy()
        __bestProb=__bestProb[__bestProb.l_identification>self.zeroThreshold]
        __bestProb=__bestProb.iloc[__bestProb.groupby(['CompNum']).p_euclidean.idxmax()]
        __numer=__bestProb[['canonicalSmiles']].copy().drop_duplicates()
        __numer['NumTmp']=range(1,len(__numer)+1)
        __bestProb=__bestProb.set_index('canonicalSmiles',drop=False).join(__numer.set_index('canonicalSmiles')).set_index('NumTmp',drop=False).sort_index()
        __bestProb['CompNum']=__bestProb['NumTmp']
        __bestProb=__bestProb.drop(columns=['NumTmp']).reset_index(drop=True)

        df=__bestProb[__bestProb.identificationApproach=='non-targeted-screening'].copy().reset_index(drop=True)
        df=df.iloc[df.groupby(['CompNum']).p_euclidean.idxmax()].reset_index(drop=True)
        __bestProb=pd.concat([__bestProb[__bestProb.identificationApproach!='non-targeted-screening'],df],sort=False,ignore_index=True)
        __bestProb=__bestProb.sort_values(['CompNum','p_euclidean']).reset_index(drop=True)
        del df

        if isinstance(compNum,int):
            __compNum=max(compNum%(__bestProb.CompNum.max()+1),1)
            __bestProb=__bestProb[__bestProb.CompNum==__compNum]

        if saveResults: __bestProb.to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsBestProbability.tsv"),sep="\t",index=False)
        return __bestProb


    def getBudget(self,compNum=None,areBest=True,saveResults=True):

        if self.__identificandsComponents.empty: return
        
        if areBest:
            __probs=self.getIdentificandsBestNaiveProbability()
        else:
            __probs=self.getIdentificandsNaiveProbabilityCombination()

        __probs=__probs[__probs.columns[~((__probs.columns.str.contains("Effect_")) | (__probs.columns.str.contains("Contribution_")))]]
            
            
        __ident=__probs[['CompNum','canonicalSmiles','name','identificationApproach','detectionCompNum','Imputed','l_identification']].copy()    
        __budgetComps=__probs.T[np.bool_(__probs.columns.str.find("l_").to_numpy()+1)].T.copy()
        __lIdentification=__budgetComps['l_identification'].copy()
        __budgetComps=__budgetComps.drop(columns=['l_identification'])
        __budgetComps=__budgetComps.apply(lambda x: 1-x)
        __pSum=np.reshape(100.0/__budgetComps.T.sum().to_numpy(),(len(__budgetComps),1))
        __budgetComps=__ident.join(pd.DataFrame(__budgetComps.to_numpy()*__pSum, columns=['contribution_'+col for col in __budgetComps.columns]))

        if isinstance(compNum,int):
            __compNum=max(compNum%(__budgetComps.CompNum.max()+1),1)
            __budgetComps=__budgetComps[__budgetComps.CompNum==__compNum]

        if saveResults: __budgetComps.to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsIdentificationBudget.tsv"),sep="\t",index=False)
        
        return (__budgetComps,__probs)
        


    def getEffectsAndContributions(self,probsComps):

        __probs=probsComps
        __compInfo=__probs[['name','identificationApproach','CompNum','l_identification']].copy()
        __budgetComps=__probs.filter(regex='^l_')
        __budgetComps=__budgetComps.drop(columns=['l_identification'])

        __metrics=["l_identification","p_euclidean","p_Lp_2.5","p_Lp_1.5","p_hellinger","p_area"]
        __nMetrics=len(__metrics)
        __effects=np.zeros((__nMetrics,)+np.shape(__budgetComps))
        __contributions=np.zeros((__nMetrics,)+np.shape(__budgetComps))

        __pRef=__probs[__metrics].to_numpy()
        for idx,comp_ in enumerate(__budgetComps.columns):
            __q_contributions=__budgetComps.copy().astype(float).to_numpy()
            __q_effects=np.ones(np.shape(__budgetComps))
            __q_effects[:,idx]=__q_contributions[:,idx]
            __q_contributions[:,idx]=1.0

            __contributions[0,:,idx]=self.p_likehood(__q_contributions)
            __effects[0,:,idx]=np.abs(__pRef[:,0]-__contributions[0,:,idx])/__contributions[0,:,idx]*100

            #p euclidean    
            __contributions[1,:,idx]=self.p_Lp(__q_contributions)
            __effects[1,:,idx]=np.abs(__pRef[:,1]-__contributions[1,:,idx])/__contributions[1,:,idx]*100

            #p Lp 2.5
            __contributions[2,:,idx]=self.p_Lp(__q_contributions,2.5)
            __effects[2,:,idx]=np.abs(__pRef[:,2]-__contributions[2,:,idx])/__contributions[2,:,idx]*100

            #p Lp 1.5
            __contributions[3,:,idx]=self.p_Lp(__q_contributions,1.5)
            __effects[3,:,idx]=np.abs(__pRef[:,3]-__contributions[3,:,idx])/__contributions[3,:,idx]*100

            #p Hellinger
            __contributions[4,:,idx]=self.p_hellinger(__q_contributions)
            __effects[4,:,idx]=np.abs(__pRef[:,4]-__contributions[4,:,idx])/__contributions[4,:,idx]*100

            #p area
            __contributions[5,:,idx]=self.p_area_trapezoid(__q_contributions)
            __effects[5,:,idx]=np.abs(__pRef[:,5]-__contributions[5,:,idx])/__contributions[5,:,idx]*100



        for idx in range(np.shape(__contributions)[0]):
            __contributions[idx,:,:]=np.abs(1.0-__pRef[:,idx][:, np.newaxis]/__contributions[idx,:,:])
            __contributions[idx,:,:]=__contributions[idx,:,:]/np.sum(__contributions[idx,:,:],axis=1)[:, np.newaxis]*100

        compDF=pd.DataFrame()
        for idx in range(len(__metrics)):
            __labelsE=np.repeat(("Effect."+__metrics[idx]+"_"),__budgetComps.shape[1])+__budgetComps.columns
            __labelsE=[s.replace("_l_", "_") for s in __labelsE]
            __labelsC=[s.replace("Effect.", "Contribution.") for s in __labelsE]
            if compDF.empty:
                compDF=pd.DataFrame(__effects[idx],columns=__labelsE).join(pd.DataFrame(__contributions[idx],columns=__labelsC))
            else:
                compDF=compDF.join(pd.DataFrame(__effects[idx],columns=__labelsE).join(pd.DataFrame(__contributions[idx],columns=__labelsC)))

        compDF.columns=[s.replace("Effect.", "Effect_").replace("Contribution.", "Contribution_") for s in compDF.columns]

        
        del __contributions
        del __effects
        return compDF


        


    
        
    def plotEffectsByCompound(self,compNum=0,areBest=True,saveFig=False,withoufFiltering=False):

        if self.__identificandsComponents.empty: return
        
        if areBest:
            __probs=self.getIdentificandsBestNaiveProbability()
        else:
            __probs=self.getIdentificandsNaiveProbabilityCombination()
            if withoufFiltering:
                __probs=pd.read_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsProbabilityWithoutFiltering.tsv"),sep="\t")

        __probs=__probs[__probs.columns[~((__probs.columns.str.contains("Effect_")) | (__probs.columns.str.contains("Contribution_")))]]

        __compNum=max(compNum%(__probs.CompNum.max()+1),1)
        __compInfo=__probs[['name','identificationApproach','CompNum','l_identification']].copy()
        __budgetComps=__probs.T[np.bool_(__probs.columns.str.find("l_").to_numpy()+1)].T.copy()
        __budgetComps=__budgetComps.drop(columns=['l_identification'])
        __budgetComps=__budgetComps.apply(lambda x: (1-x)*100)
        __budgetComps['CompNum']=__compInfo.CompNum
        __budgetComps=__budgetComps[__budgetComps.CompNum==__compNum].drop(columns=['CompNum'])
        __compInfo=__compInfo[__compInfo.CompNum==__compNum]
        __compsIdx=list(__probs.columns.str.find("l_chromSignalQlty")).index(0)
        __compProbs=__probs[__probs.CompNum==__compNum].copy().iloc[:,__compsIdx:__compsIdx+6]

        width = 1/(len(__budgetComps)*1.5)
        __contribs=[lbl.replace("l_","").replace("(","").replace(")","") for lbl in __budgetComps.columns]
        x = np.arange(len(__contribs))

        plt.rcdefaults()
        plt.clf()
        fig, ax = plt.subplots(layout='constrained')
        for idxComp,compound in enumerate(range(len(__budgetComps))):
            offset = width * idxComp

            __valsLabel=[ f"{val[0]:.2F} % ({val[1]:.3F})" for val in zip(__budgetComps.iloc[idxComp,:],__compProbs.iloc[idxComp,:]) ]
            rects = ax.barh(x + offset,__budgetComps.iloc[idxComp,:], width, label=f"{__compInfo.name.iloc[idxComp]} ({__compInfo.identificationApproach.iloc[idxComp]}): l={__compInfo.l_identification.iloc[idxComp]:.3F}")
            ax.bar_label(rects,labels=__valsLabel, padding=5,fontsize=10*21/(7*len(__budgetComps)))


        ax.set_xlabel('% Contribution Effect')
        ax.set_title('Effects')
        ax.set_yticks(x + width, __contribs)
        ax.legend(bbox_to_anchor=(1.0,min(-0.3,-0.225*(7*len(__budgetComps))/21)),loc='lower right', ncols=1,fontsize=8, borderaxespad=0.)
        
        ax.set_xlim(0, np.ceil(np.array(__budgetComps).max()*1.35))

        if saveFig:
            plt.savefig(f"{__compInfo.iloc[0]['name']}_lh_effects.pdf", bbox_inches='tight')

        plt.show()
        plt.clf()
        return 


    def __barPlot(self,ax,budgetComps,compInfo,l_comps,maxLim):
        width = 1/1.5
        __contribs=[lbl.replace("contribution_l_","").replace("(","").replace(")","") for lbl in budgetComps.index]
        x = np.arange(len(budgetComps))
        __valsLabel=[ f"{val[0]:.2F} % ({val[1]:.3F})" for val in zip(budgetComps,l_comps)]
        rects = ax.barh(x+(width),budgetComps, width)
        ax.bar_label(rects,labels=__valsLabel, padding=5)
        ax.set_xlabel('% Uncertainty Contribution')
        ax.set_title(f"{compInfo['name']} ({compInfo.identificationApproach}): l={compInfo.l_identification:.3F}")
        ax.set_yticks(x + width, __contribs)
        ax.set_xlim(0, np.ceil(maxLim*1.15))

    def plotIonBudget(self,compNum=0,areBest=True,cols=1,figsize=(20,10),saveFig=False):
        if self.__identificandsComponents.empty: return
        pb=self.getBudget(areBest=areBest)
        __budgetComps=pb[0].copy()
        
        __lhComps=pb[1].copy()
        __compNum=max(compNum%(len(__budgetComps)+1),1)
        __budgetComps=__budgetComps[__budgetComps.CompNum==__compNum].sort_values(['l_identification'],ascending=False).reset_index(drop=True)

        __compsIdx=list(__lhComps.columns.str.find("l_chromSignalQlty")).index(0)
        __lhComps=__lhComps[__lhComps.CompNum==__compNum].sort_values(['l_identification'],ascending=False).reset_index(drop=True).iloc[:,__compsIdx:__compsIdx+6]
        __compInfo=__budgetComps[['name','identificationApproach','CompNum','l_identification']].copy()
        __budgetComps=__budgetComps.T[np.bool_(__budgetComps.columns.str.find("l_").to_numpy()+1)].T.drop(columns=['l_identification'])
        __maxLim=__budgetComps.max().max()
        rows=int(np.ceil(len(__compInfo)/cols))

        plt.rcdefaults()
        plt.clf()
        fig, ax = plt.subplots(rows,cols,layout='constrained',figsize=figsize,sharey=True,squeeze=False)

        cidx=-1
        for idxRows in range(rows):
            for idxCols in range(cols):
                cidx+=1
                if cidx<len(__compInfo):
                    self.__barPlot(ax[idxRows][idxCols],__budgetComps.iloc[cidx],__compInfo.iloc[cidx],__lhComps.iloc[cidx],__maxLim)


        fig.suptitle(f"Percentage Contributions - Compound: {compNum}",fontsize=20)
        if saveFig:
            plt.savefig(f"{__compInfo.iloc[0]['name']}_lh_contributions.pdf")
        plt.show()
        plt.clf()
        return 



    def __plotEffects(self,assessComponents,identifLikelihood,compInfo,metric,strReplacement,saveFig=False):
        width = 1/(len(assessComponents)*1.5)
        __contribs=[lbl.replace(strReplacement+"_","").replace("(","").replace(")","") for lbl in assessComponents.columns]
        x = np.arange(len(__contribs))
        plt.rcdefaults()
        plt.clf()
        fig, ax = plt.subplots(layout='constrained')

        for idxComp,compound in enumerate(range(len(assessComponents))):
            offset = width * idxComp + width/(len(identifLikelihood))

            __valsLabel=[ f"{val[0]:.2F} % ({val[1]:.3F})" for val in zip(assessComponents.iloc[idxComp,:],identifLikelihood.iloc[idxComp,:]) ]
            rects = ax.barh(x + offset,assessComponents.iloc[idxComp,:], width, label=f"{compInfo.name.iloc[idxComp]} ({compInfo.identificationApproach.iloc[idxComp]}): {metric}={compInfo[metric].iloc[idxComp]:.3F}")
            ax.bar_label(rects,labels=__valsLabel, padding=5,fontsize=10/(1+np.emath.logn(5,len(assessComponents))))


        ax.set_xlabel('% Contribution Effect')
        ax.set_title('Identificand Effects')
        ax.set_yticks(x + width, __contribs)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncols=1,fontsize=8, borderaxespad=0.)

        ax.set_xlim(0, np.ceil(np.array(assessComponents).max()*1.4))

        if saveFig:
            plt.savefig(f"{compInfo.iloc[0]['name']}_effects.pdf", bbox_inches='tight')

        plt.show()
        plt.clf()
        return 

    def __plotProbabilityComponents(self,identifLikelihood,compInfo,metric,saveFig=False):
        width = 1/(len(identifLikelihood)*1.5)
        __contribs=[lbl.replace("l_","").replace("(","").replace(")","") for lbl in identifLikelihood.columns]
        x = np.arange(len(__contribs))
        plt.rcdefaults()
        plt.clf()
        fig, ax = plt.subplots(layout='constrained')
        for idxComp,compound in enumerate(range(len(identifLikelihood))):
            offset = width * idxComp + width/(len(identifLikelihood))

            __valsLabel=[ f"{val:.3F}" for val in identifLikelihood.iloc[idxComp,:] ]
            rects = ax.barh(x + offset,identifLikelihood.iloc[idxComp,:], width, label=f"{compInfo.name.iloc[idxComp]} ({compInfo.identificationApproach.iloc[idxComp]}): {metric}={compInfo[metric].iloc[idxComp]:.3F}")
            ax.bar_label(rects,labels=__valsLabel, padding=5,fontsize=10/(1+np.emath.logn(5,len(__contribs))))

        ax.set_xlabel('probability')
        ax.set_title('Identification Components')
        ax.set_yticks(x + width, __contribs)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncols=1,fontsize=8, borderaxespad=0.)
        
        ax.set_xlim(0, np.ceil(np.array(identifLikelihood).max()*1.35))

        if saveFig:
            plt.savefig(f"{compInfo.iloc[0]['name']}_probabilityComponents.pdf", bbox_inches='tight')

        plt.show()
        plt.clf()
        return 


    def __plotContributionsBudget(self,compNum,indentContributions,compInfo,lhComps,metric,strReplacement,figsize=(15,15),saveFig=False):
        width = 1/1.5
        rows=len(indentContributions)
        __contribs=indentContributions.copy()
        __contribs.columns=[lbl.replace(strReplacement+"_","").replace("(","").replace(")","") for lbl in indentContributions.columns]
        plt.rcdefaults()
        plt.clf()
        fig, axs = plt.subplots(rows,1,layout='constrained',figsize=figsize,sharey=True,squeeze=False)
        maxLim=indentContributions.max().max()

        for idxRows,ax in enumerate(axs.flat):
            x = np.arange(len(__contribs.iloc[idxRows]))
            __valsLabel=[ f"{val[0]:.2F} % ({val[1]:.3F})" for val in zip(__contribs.iloc[idxRows],lhComps.iloc[idxRows])]
            rects = ax.barh(x+(width),__contribs.iloc[idxRows], width)
            ax.bar_label(rects,labels=__valsLabel, padding=5)
            ax.set_xlabel('% Contribution')
            ax.set_title(f"{compInfo['name'].iloc[idxRows]} ({compInfo.iloc[idxRows].identificationApproach}): {metric}={compInfo.iloc[idxRows][metric]:.3F}")
            ax.set_yticks(x + width, __contribs.columns)
            ax.set_xlim(0, np.ceil(maxLim*1.1))

        fig.suptitle(f"Percentage Contributions to Decreased {metric} - Compound: {compNum}",fontsize=20)


        if saveFig:
             plt.savefig(f"{compInfo.iloc[0]['name']}_contributions.pdf", bbox_inches='tight')
        plt.show()
        plt.clf()
        return 


    def plotProbabilityBudgets(self,compNum=0,metric="p_euclidean",assessType="Effect",areBest=True,approach=None,cols=1,figsize=(20,10),saveFig=False,
                               withoufFiltering=False):
        if self.__identificandsComponents.empty: return
        
        __probs=self.getIdentificandsNaiveProbabilityCombination()
        if withoufFiltering:
            __probs=pd.read_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_identificandsProbabilityWithoutFiltering.tsv"),sep="\t")
        
        __assessType=assessType.capitalize()+"_"+metric.lower()

        if __probs.columns.str.find(__assessType).any():
            __compNum=max(compNum%(__probs.CompNum.max()+1),1)
            __compInfo=__probs[['name',"inchikey",'identificationApproach','CompNum',metric]].copy()


            __budgetComps=(__probs.T[np.bool_(__probs.columns.str.find("l_").to_numpy()+1)].T.copy()).drop(columns=['l_identification'])
            __budgetComps=__budgetComps[__budgetComps.columns[~((__budgetComps.columns.str.contains("Effect_")) | (__budgetComps.columns.str.contains("Contribution_")))]]

            __inchikey=__compInfo[__compInfo.CompNum==__compNum].inchikey
            __candidates=__compInfo[__compInfo.inchikey==__inchikey.iloc[0]]
            if "non-targeted-screening" in __candidates.identificationApproach.to_list():
                ucid=__candidates[__candidates.identificationApproach=="non-targeted-screening"].CompNum.iloc[0]
                __candidates=pd.concat([__candidates,__compInfo[__compInfo.CompNum==ucid]],sort=False,ignore_index=False).drop_duplicates()


            if not(isinstance(approach,type(None))):
                if approach in ["targeted-analysis","suspect-screening","non-targeted-screening"]:
                    __candidates=__candidates[__candidates.identificationApproach==approach]
                    if __candidates.empty: return
                else: return
                
            __concernIdx=__candidates.index
            if areBest:
                __concernIdx=__candidates.groupby("identificationApproach")[metric].idxmax().to_list()

            __probsAssessment=__probs.loc[:,~np.bool_(__probs.columns.str.find(__assessType))]
            __probsAssessment=__probsAssessment.loc[__concernIdx]
            __budgetComps=__budgetComps.loc[__concernIdx]
            __compInfo=__compInfo.loc[__concernIdx]

        
            if assessType.lower()=="effect":
                self.__plotEffects(__probsAssessment,__budgetComps,__compInfo,metric,__assessType,saveFig=saveFig)
            elif assessType.lower()=="contribution":
                self.__plotContributionsBudget(__compNum,__probsAssessment,__compInfo,__budgetComps,metric,__assessType,saveFig=saveFig)            
            else:
                self.__plotProbabilityComponents(__budgetComps,__compInfo,metric,saveFig=saveFig)

        return


    def getSummaryInfo(self,metric="p_euclidean",dtRThres=0.35,indentificandThres=0.45,ptRThres=0.50,areBest=True):

        __naiveCombinationComps=self.getIdentificandsNaiveProbabilityCombination(saveResults=False)
        __naiveCombinationComps=__naiveCombinationComps[__naiveCombinationComps.l_identification>self.zeroThreshold]
        
        if __naiveCombinationComps.empty: return
            
        if not(metric in  __naiveCombinationComps.columns): metric="p_euclidean"
        __naiveCombinationComps=__naiveCombinationComps[__naiveCombinationComps.Imputed==False]
        __blankFile=glob.glob(self.blankPath+f"/*_identificandsProbability.tsv")
        if len(__blankFile)>0:
            __blankResults=pd.read_csv(__blankFile[0],sep="\t")
            __blankResults=__blankResults[__blankResults.Imputed==False]
            commons=set(__blankResults.inchikey).intersection(set(__naiveCombinationComps.inchikey))
            sample_commons=(__naiveCombinationComps.set_index("inchikey",drop=False).loc[list(commons)])[['inchikey','intensity','tR']].drop_duplicates()
            blank_commons=(__blankResults.set_index("inchikey",drop=False).loc[list(commons)])[['inchikey','intensity','tR']].drop_duplicates()
            blank_commons=blank_commons.join(sample_commons,rsuffix="_sample",lsuffix="_blank").reset_index(drop=True)
            blank_commons['dtR']=np.abs(blank_commons.tR_sample-blank_commons.tR_blank)
            blank_commons['dInt']=(blank_commons.intensity_sample-blank_commons.intensity_blank)/blank_commons.intensity_blank
            inchi=(blank_commons[((blank_commons.dInt<=self.__dIntThres) & (blank_commons.dtR<=dtRThres))])[['inchikey_blank']].inchikey_blank.to_list()
            inchiUnknowns=__naiveCombinationComps.set_index("inchikey").loc[inchi]
            inchiUnknowns=inchiUnknowns[inchiUnknowns.identificationApproach=='non-targeted-screening'].drop_duplicates().CompNum.to_list()
            inchi=np.unique(inchi+__naiveCombinationComps.set_index("CompNum").loc[inchiUnknowns].inchikey.drop_duplicates().to_list())
            inchi=list(set(__naiveCombinationComps.inchikey)-set(inchi))
            __naiveCombinationComps=__naiveCombinationComps.set_index("inchikey",drop=False).loc[inchi].reset_index(drop=True)


        __naiveCombinationComps.insert(15,"p_tR_inference",__naiveCombinationComps.l_tR_inference/__naiveCombinationComps.w_tR)
        __naiveCombinationComps.to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_summaryResultsWithoutFiltering.tsv"),sep="\t",index=False)
        __naiveCombinationComps=__naiveCombinationComps[(__naiveCombinationComps.l_ms2>=indentificandThres) &
                            (__naiveCombinationComps.l_amenability>=indentificandThres) &
                            (__naiveCombinationComps['l_ionizedSpecies']>=indentificandThres) &
                            (__naiveCombinationComps.p_tR_inference>=indentificandThres) &
                            (__naiveCombinationComps.l_isotopicProfile>=indentificandThres) ].reset_index(drop=True)

        __naiveCombinationComps=__naiveCombinationComps[(__naiveCombinationComps.p_tR_inference>=ptRThres)].reset_index(drop=True)

        
        if areBest:
            __naiveCombinationComps=__naiveCombinationComps.iloc[__naiveCombinationComps.groupby("CompNum").p_euclidean.idxmax().to_list()].reset_index(drop=True)

        colsToSelect=["compID","CompNum","canonicalSmiles","inchikey","name","bmonoisotopicMass","identificationApproach","intensity","tR","tR_inference",
                  "U_tR_inference","p_tR_inference","massError_ppm",metric,"l_identification",
                  "l_chromSignalQlty","l_isotopicProfile","l_amenability","l_tR_inference","l_ionizedSpecies","l_ms2",
                  "p_candidate","candidatesNum"]+(__naiveCombinationComps.loc[:,list(__naiveCombinationComps.columns.str.find(metric)>=0)].drop(columns=metric)).columns.to_list()

        __naiveCombinationComps=__naiveCombinationComps.sort_values(metric,ascending=False)
        groupID=__naiveCombinationComps[["inchikey"]].drop_duplicates().reset_index(drop=True)
        groupID['compID']=range(1,len(groupID)+1)
        __naiveCombinationComps=groupID.set_index("inchikey").join(__naiveCombinationComps.set_index("inchikey",drop=False)).reset_index(drop=True)
        __naiveCombinationComps.to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_summaryResultsWithAllMetrics.tsv"),sep="\t",index=False)
        __naiveCombinationComps=__naiveCombinationComps[colsToSelect]
        __naiveCombinationComps.to_csv(os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_summaryResults.tsv"),sep="\t",index=False)
        return __naiveCombinationComps



    
