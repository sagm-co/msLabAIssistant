import sys,os
from pathlib import Path
import time
from datetime import datetime
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

from IPython.display import display, clear_output
from tqdm import tqdm
#pd.options.mode.copy_on_write = True 

sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from lcmsMeasurements import lcmsMeasurements
from msMultiCompounds import msMultiCompounds
from msMLModelsCommon import msMLModelsCommon
from msChemicalFormulasInference import msChemicalFormulasInference

msMLModelsComm=msMLModelsCommon()

class msTargetAnalysis(lcmsMeasurements,msChemicalFormulasInference):
    def __init__(self):
        lcmsMeasurements.__init__(self)
        msChemicalFormulasInference.__init__(self)
        self.__msCompoundsManager=msMultiCompounds()        
        self.isotopicSignals_mzErr=[5.0,5.0]
        self.__detected=0
        self.__removeFiles=False

        
        ##Current compound assessment
        self.__currentIsotopicPatternSection=pd.DataFrame()
        self.__currentIsolatedIsotopicPattern=pd.DataFrame()
        self.__currentXICPeakAssessment=pd.DataFrame()
        self.__currentCompoundIdx=-1
        self.__currentInteractionProductIdx=-1
        self.__currPolarity="+"
        self.__pCurrentCompound_Amenbility=1.0
        self.__currentCompound_tR=None
        self.__tRProbThreshold=0.0

        ##Compiled info for compounds found
        self.__allDetectedCompounds=pd.DataFrame()
        self.__currentDetectedCompound=pd.DataFrame()
        self.__isotopicPattenSectionsForCompoundsFound=pd.DataFrame()
        self.__isotopicPattensForCompoundsFound=pd.DataFrame()
        self.__XICsForCompoundsFound=pd.DataFrame()
        self.__numOfdetectedCompounds=0
        self.__ionsDetectionFile=""
        self.__targetXICsFile=""
        self.__targetFragmentsXICsFile=""
        self.__targetFragmentsIonsFoundFile=""
        self.__targetFragmentsIonsFoundAnnotatedFile=""
        self.__targetIsotopicPatternsFile=""
        self.__elapsedTimeFile=""

        ##Search parameters
        self.__mzAcc_ppm=[5.0,5.0]
        self.__intensityThresError=[0.35,0.35] 
        self.__tR_error=0.1
        self.numOfIsoSignals=1
        self.__numOfIsoSignalsFilter=self.numOfIsoSignals
        self.__compoundXICSignalMinNumOfPoints=5

        ##Adquisition info
        self.__rawFile=None
        self.__targetCompoundProperties=pd.DataFrame()
        self.__filters=pd.DataFrame()
        self.__FSFilters=pd.DataFrame()       
        self.__resultsPath="./resultsData"
        self.__bkresultsPath="./resultsData"
        self.__madeDefaultResultsDirectory=False
        self.__rawFileBaseName=""
        self.__targetMode=""
        self.__isTargetKnowns=False


    def __iter__(self):
        return self

    def __next__(self):
              
        self.__currentInteractionProductIdx=(self.__currentInteractionProductIdx+1)%self.__msCompoundsManager.interactionFormulas.numOfInteractionProducts
            
        if ( (self.__currentInteractionProductIdx==0) & (self.__currentCompoundIdx==(self.__msCompoundsManager.numOfSmilesFormulas-1) ) ):
            self.__currentInteractionProductIdx=-1
            self.__currentCompoundIdx=-1
            raise StopIteration
        else:               

            if self.__currentInteractionProductIdx==0:
                self.__currentCompoundIdx=(self.__currentCompoundIdx+1)%self.__msCompoundsManager.numOfSmilesFormulas

            self.__msCompoundsManager.currentCompoundIndex=self.__currentCompoundIdx            
            self.__msCompoundsManager.interactionFormulas.interactionProductIndex=self.__currentInteractionProductIdx


    @property
    def interactionProductIndexes(self):
        return(self.__currentCompoundIdx,self.__currentInteractionProductIdx)
        
    @interactionProductIndexes.setter
    def interactionProductIndexes(self,value):
        self.__currentCompoundIdx=value[0]
        self.__currentInteractionProductIdx=value[1]

        self.__msCompoundsManager.currentCompoundIndex=self.__currentCompoundIdx            
        self.__msCompoundsManager.interactionFormulas.interactionProductIndex=self.__currentInteractionProductIdx

    @property
    def removeFiles(self):
        return self.__removeFiles

    @removeFiles.setter
    def removeFiles(self,value):
        self.__removeFiles=value


    @property
    def targetMode(self):
        return self.__targetMode

    @targetMode.setter
    def targetMode(self,value):
        self.__targetMode=value

        
    @property
    def rawFile(self):
        return self.__rawFile

    @rawFile.setter
    def rawFile(self,value):
        self.__allDetectedCompounds=pd.DataFrame()
        self.__rawFile=str(Path(value).resolve())
        self.MSfile=self.__rawFile
        self.events=self.__rawFile
        self.__getFilters()
        self.currMS1ScanNum2tRmax=None
        self.__currentCompoundIdx=-1
        self.__currentInteractionProductIdx=-1
        self.__rawFileBaseName=os.path.basename(self.__rawFile).replace('.raw','')


        if not(os.path.exists(self.__bkresultsPath)):
            os.mkdir(self.__bkresultsPath)

        self.__resultsPath=os.path.join(self.__bkresultsPath,self.__rawFileBaseName)
        if not(os.path.exists(self.__resultsPath)):
            os.mkdir(self.__resultsPath)

        #FS XICs
        self.__targetXICsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_{self.targetMode}FSXICs.tsv")
        if os.path.exists(self.__targetXICsFile) & self.__removeFiles: os.remove(self.__targetXICsFile)

        self.__targetIsotopicPatternsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_{self.targetMode}IsotopicPatterns.tsv")
        if os.path.exists(self.__targetIsotopicPatternsFile) & self.__removeFiles: os.remove(self.__targetIsotopicPatternsFile)

        #Compound detection info
        self.__ionsDetectionFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_{self.targetMode}AllDetectedCompounds.tsv")
        if os.path.exists(self.__ionsDetectionFile) & self.__removeFiles: os.remove(self.__ionsDetectionFile)

        #Fragments data
        self.__targetFragmentsXICsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_{self.targetMode}XICs4fragmentsIonsFound.tsv")
        if os.path.exists(self.__targetFragmentsXICsFile) & self.__removeFiles: os.remove(self.__targetFragmentsXICsFile)

        self.__targetFragmentsIonsFoundFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_{self.targetMode}FragmentsIonsFound.tsv")
        if os.path.exists(self.__targetFragmentsIonsFoundFile) & self.__removeFiles: os.remove(self.__targetFragmentsIonsFoundFile)

        self.__targetFragmentsIonsFoundAnnotatedFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_{self.targetMode}FragmentsIonsFoundAnnotated.tsv")
        if os.path.exists(self.__targetFragmentsIonsFoundAnnotatedFile) & self.__removeFiles: os.remove(self.__targetFragmentsIonsFoundAnnotatedFile)

        #elapsed time
        self.__elapsedTimeFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_{self.targetMode}Identification_elapsedTime.tsv")
        if os.path.exists(self.__elapsedTimeFile) & self.__removeFiles: os.remove(self.__elapsedTimeFile)

        
    @property
    def resultsPath(self):
        return self.__resultsPath

    @resultsPath.setter
    def resultsPath(self,value):
        if self.__bkresultsPath!=value:
            if len(os.listdir(self.__resultsPath))==0:
                os.remove(self.__resultsPath)
                if len(os.listdir(self.__bkresultsPath))==0:
                    os.remove(self.__bkresultsPath)

            self.__resultsPath=value
            self.__bkresultsPath=value            
            if not(os.path.exists(self.__resultsPath)):
                os.mkdir(self.__resultsPath)
                if self.__rawFileBaseName!="":
                    self.__resultsPath=os.path.join(self.__bkresultsPath,self.__rawFileBaseName)
                    os.mkdir(self.__resultsPath)

                
            
    @property
    def targetCompounds(self):
        __targetComps=self.__msCompoundsManager.smilesFormulas.join(self.__msCompoundsManager.ionSpecies)
        return __targetComps

    @targetCompounds.setter
    def targetCompounds(self,value):

        if isinstance(value,str):
            if os.path.exists(value):
                self.__targetCompoundProperties=pd.read_csv(value,sep="\t")
                self.__targetCompoundProperties=self.__targetCompoundProperties[ (self.__targetCompoundProperties.smiles!="x") & (self.__targetCompoundProperties.smiles!="X")].reset_index(drop=True)
                self.__targetCompoundProperties=self.__targetCompoundProperties[~self.__targetCompoundProperties.smiles.isna()]
        else:
            self.__targetCompoundProperties=value
            self.__targetCompoundProperties=self.__targetCompoundProperties[ (self.__targetCompoundProperties.smiles!="x") & (self.__targetCompoundProperties.smiles!="X")].reset_index(drop=True)
            self.__targetCompoundProperties=self.__targetCompoundProperties[~self.__targetCompoundProperties.smiles.isna()]

        if self.__targetCompoundProperties.empty: return
        
        if "analyze" in self.__targetCompoundProperties.columns:
            self.__targetCompoundProperties=self.__targetCompoundProperties[self.__targetCompoundProperties['analyze']!=False]
            self.__targetCompoundProperties=self.__targetCompoundProperties.drop(columns=['analyze']).reset_index(drop=True)
            
        self.__msCompoundsManager.smilesFormulas=self.__targetCompoundProperties
       
    
    @property
    def targetCompoundProperties(self):
        return self.__targetCompoundProperties
        
    @property
    def interationFormulas(self):
        return self.__msCompoundsManager.interactionFormulas.interactionProducts

    @interationFormulas.setter
    def interationFormulas(self,value):
        self.__msCompoundsManager.interactionFormulas.interactionProducts=value

    @property
    def compoundXICSignalMinNumOfPoints(self):
        return self.__compoundXICSignalMinNumOfPoints

    @compoundXICSignalMinNumOfPoints.setter
    def compoundXICSignalMinNumOfPoints(self,value):
        self.__compoundXICSignalMinNumOfPoints=value

    @property
    def totalSignalToSearch(self):
        return self.__msCompoundsManager.totalIons

    @property
    def currentDetectedCompound(self):
        return self.__currentDetectedCompound
    
    @property
    def currentTargetCompound(self):
        return self.__msCompoundsManager.interactionFormulas

    @property
    def allDetectedCompounds(self):
        return self.__allDetectedCompounds
    
    @property
    def mzAcc_ppm(self):
        return self.__mzAcc_ppm

    @mzAcc_ppm.setter
    def mzAcc_ppm(self):
        self.__mzAcc_ppm=value

    @property
    def intensityThresError(self):
        if len(self.__intensityThresError)<self.currentTargetCompound.theorIsoPatternPeaksNum:
            self.intensityThresError=self.__intensityThresError
            
        return self.__intensityThresError
        
    @intensityThresError.setter
    def intensityThresError(self,value):
        value_=value
        if not(isinstance(value,list)):value_=[value]
        npeaks=self.currentTargetCompound.theorIsoPatternPeaksNum-1
        self.__intensityThresError=[0.01]+value_[0:min(len(value_),npeaks)]+[value_[-1]]*(npeaks-min(len(value_),npeaks+1)) 
        

    @property
    def tRError(self):
        return self.__tR_error
        
    @tRError.setter
    def tRError(self,value):
        self.__tR_error=value

    @property
    def instrumentalMethod(self):
        return msMLModelsComm.instrumentalMethod

    @instrumentalMethod.setter
    def instrumentalMethod(self,value):
        msMLModelsComm.instrumentalMethod=value
        
    @property
    def currentIsotopicPatternSection(self):
        return self.__currentIsotopicPatternSection

    @property
    def currentIsolatedIsotopicPattern(self):
        return self.__currentIsolatedIsotopicPattern

    @property
    def currentXICPeakAssessment(self):
        return self.__currentXICPeakAssessment


    @property    
    def currentTheoreticalIsotopicPattern(self):
        return self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern

    
    def __getFilters(self):
        self.__filters=self.getFilters(self.__rawFile)
        self.__FSFilters=self.__filters.copy()
        self.__FSFilters=self.__FSFilters[self.__FSFilters['MSn']=="ms1"]


    def __fragmentsIonsAssessment(self):
        self.__currentDetectedFragments=pd.DataFrame()
        if(self.probableFragmentIons.empty): return

        __ms4annotation=self.probableFragmentIons[['mz','intensity']].copy()
        __currentCompound=self.__msCompoundsManager.interactionFormulas
        self.__msCompoundsManager.interactionFormulas.msFragmenter.msSpectrum4Annotation=__ms4annotation
        __extDB=pd.DataFrame({'MonoisotopicMass':[__currentCompound.baseExactMass.iloc[0].exactMW],
                              'Identifier':[f"{self.__currentCompoundIdx}:{self.__currentInteractionProductIdx}"],
                              'MolecularFormula':[__currentCompound.baseFormula],
                              'SMILES':[__currentCompound.smiles],
                              'InChI':[__currentCompound.inchikey]})


        __db_name=f"db_d{self.__currentCompoundIdx}_{datetime.now().strftime('%y%m%d%H%M%S%f')}"
        __extDB.to_csv(__db_name,sep=",",index=False)
        __currentCompound.msFragmenter.annotateSpectrum(__currentCompound.smiles,
                                                                       __currentCompound.currentIonSpecies[0],
                                                                       __currentCompound.baseFormula,
                                                                       __db_name,
                                                                       __extDB.MonoisotopicMass.iloc[0]
                                                                      )
        os.remove(__db_name)


    def __saveDectectionInfo(self,saveOutputs=True):
        
        if self.probableFragmentIons.empty: return
        
        #Save fragments' XICs
        __ionNum=f"{self.__currentCompoundIdx}:{self.__currentInteractionProductIdx}"
        __currentDetectedFragments=self.probableFragmentIonsXICs.copy()
        __currentDetectedFragments.insert(0,'CompNum',self.__numOfdetectedCompounds)
        __currentDetectedFragments.insert(1,'IonNum',__ionNum)        
        if (os.path.exists(self.__targetFragmentsXICsFile)):
            __currentDetectedFragments.to_csv(self.__targetFragmentsXICsFile,sep="\t",index=False, mode='a',header=False)
        else:
            __currentDetectedFragments.to_csv(self.__targetFragmentsXICsFile,sep="\t",index=False)

        #Save fragment ions
        __currentDetectedFragments=self.probableFragmentIons.copy()
        __currentDetectedFragments.insert(0,'CompNum',self.__numOfdetectedCompounds)
        __currentDetectedFragments.insert(1,'IonNum',__ionNum)
        if (os.path.exists(self.__targetFragmentsIonsFoundFile)):
            __currentDetectedFragments.to_csv(self.__targetFragmentsIonsFoundFile,sep="\t",index=False, mode='a',header=False)
        else:
            __currentDetectedFragments.to_csv(self.__targetFragmentsIonsFoundFile,sep="\t",index=False)

        if not( self.__msCompoundsManager.interactionFormulas.msFragmenter.msAnnotationData.empty):
 
            self.__currentAnnotatedFragmentIons=self.__msCompoundsManager.interactionFormulas.msFragmenter.msAnnotationData.copy()
            self.__currentAnnotatedFragmentIons=self.__currentAnnotatedFragmentIons.astype({'mz':'float64'}).set_index('mz').join(self.probableFragmentIons[['mz','p_MS1_MS2','mzPrecursor']].copy().set_index('mz')).sort_values(['Score','inchikey'],ascending=False).reset_index()

            self.__currentAnnotatedFragmentIons.insert(0,'CompNum',[self.__numOfdetectedCompounds]*len(self.__currentAnnotatedFragmentIons))
            self.__currentAnnotatedFragmentIons.insert(1,'ionNum',[__ionNum]*len(self.__currentAnnotatedFragmentIons))


            self.annotatedFragmentIons=self.__currentAnnotatedFragmentIons
            if (os.path.exists(self.__targetFragmentsIonsFoundAnnotatedFile)):
                self.__currentAnnotatedFragmentIons.to_csv(self.__targetFragmentsIonsFoundAnnotatedFile,sep="\t",index=False, mode='a',header=False)
            else:
                self.__currentAnnotatedFragmentIons.to_csv(self.__targetFragmentsIonsFoundAnnotatedFile,sep="\t",index=False)

            #It adds score by ms2 metfrag's annotation
            __score=self.__currentAnnotatedFragmentIons[['IDKey','Score']].drop_duplicates().rename(columns={'Score':'p_ms2','IDKey':'idx'}).sort_values(['p_ms2'],ascending=False)['p_ms2'].iloc[0]
            self.__currentDetectedCompound['p_ms2']=__score
            
            #It adds quality score for ms2 fragments with just  metfrag's annotation
            self.__currentDetectedCompound['numMS2Fragments']=len(self.__currentAnnotatedFragmentIons)
            self.__currentDetectedCompound['p_ms2Qlty']=self.assessMS2ProfileQuality(self.__currentAnnotatedFragmentIons)

            
        else:

            #It adds quality score for all ms2 detected fragments
            n_frags=len(self.probableFragmentIons)
            self.__currentDetectedCompound['numMS2Fragments']=n_frags

            if (self.__isTargetKnowns & (n_frags>0)):
                self.__currentDetectedCompound['p_ms2']=1.0
            else:
                # There was not ms2 metfrag's annotations
                self.__currentDetectedCompound['p_ms2']=np.nan

            self.__currentDetectedCompound['p_ms2Qlty']=self.assessMS2ProfileQuality(self.probableFragmentIons)            

               
    def __analyticalApproachInferences(self):
        __properties=self.__targetCompoundProperties.iloc[[self.__currentCompoundIdx]][['ionizedSpecies','fragmentationProfile','tR','amenability']].dropna()
        if len(__properties)==0:
            __properties=self.__targetCompoundProperties.iloc[[self.__currentCompoundIdx]]           
            if __properties.amenability.isna().iloc[0]:
                p_I_A=self.__msCompoundsManager.interactionFormulas.getAmenabilityInference()
                if p_I_A.empty: self.__pCurrentCompound_Amenbility=0.0
                self.__pCurrentCompound_Amenbility=p_I_A['p'].iloc[0]
            else:
                self.__pCurrentCompound_Amenbility=__properties.amenability.iloc[0]

            if __properties.tR.isna().iloc[0]:
                self.__currentCompound_tR=self.__msCompoundsManager.interactionFormulas.getRetentionTimeInference(self.tR)
                self.__tRProbThreshold=self.tRInferenceProbThreshold
            else:
                self.__currentCompound_tR=self.__msCompoundsManager.interactionFormulas.getRetentionTimeAssessment(self.tR,__properties.tR.iloc[0],self.retentionTimeError)
                self.__tRProbThreshold=self.tRProbThreshold
                               
            return "suspect-screening"
        else:
            self.__currentCompound_tR=self.__msCompoundsManager.interactionFormulas.getRetentionTimeAssessment(self.tR,__properties.tR.iloc[0],self.retentionTimeError)
            self.__tRProbThreshold=self.tRProbThreshold
            return "targeted-analysis"


        
    def __labelIsotopicPattern(self,scanNum):
        processAsCentroid=True
        if self.MSfilter.find(" p ")>=0:
            processAsCentroid=False

        self.ms=self.getScan(scanNum,processAsCentroid)
        self.isotopicPattUpperBound=self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.mz.iloc[-1]
        self.__currentIsotopicPatternSection=self.getEIsotopicPatternSection().copy()
        self.__currentIsotopicPatternSection['ionLabel']="S"
        theorIsoPattern=self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.copy()
        theorIsoPattern.intensity=theorIsoPattern.intensity/theorIsoPattern.intensity.max()
        theorIsoPattern['ionLabel']=["M"]+["M+"+str(i) for i in range(1,len(theorIsoPattern))]
        df=pd.DataFrame(map(lambda i :self.__currentIsotopicPatternSection.iloc[np.abs(self.__currentIsotopicPatternSection.mz-theorIsoPattern.mz.iloc[i]).idxmin()],range(theorIsoPattern.shape[0])))
        df['ionLabel']=["M"]+["M+"+str(i) for i in range(1,len(theorIsoPattern))]

        df['mzError_ppm']=list(map(lambda i:(df.mz.iloc[i]-theorIsoPattern.mz.iloc[i])/theorIsoPattern.mz.iloc[i]*1E6,range(theorIsoPattern.shape[0])))
        idxMax_=theorIsoPattern.intensity[0:len(df)].idxmax()
        df['normIntensity']=df.intensity/df.intensity.iloc[idxMax_] 
        df['relativeIntensityError']=list(map(lambda i:(df.normIntensity.iloc[i]-theorIsoPattern.intensity.iloc[i])/theorIsoPattern.intensity.iloc[i],range(theorIsoPattern.shape[0])))
        mzAcc_=[self.__mzAcc_ppm[0]]+[self.__mzAcc_ppm[1]]*(len(theorIsoPattern)-1) 
        df=df[[abs(df.mzError_ppm.iloc[i])<=mzAcc_[i] for i in range(len(theorIsoPattern))]] 
        df['theorIntensity']=theorIsoPattern.intensity.to_list()[0:len(df)]

        if not(df.empty):           
            df=df.sort_values(['theorIntensity'],ascending=False) 
            intensityThresError_=self.intensityThresError
            df=df[[abs(df.relativeIntensityError.iloc[i])<=intensityThresError_[i] for i in range(len(df))]]

            if len(df)>=self.__numOfIsoSignalsFilter:
                df=df.sort_values(['mz'])
               
                for ionIdx in df.index:
                    self.__currentIsotopicPatternSection.loc[ionIdx,'ionLabel']=df.ionLabel.loc[ionIdx]

                self.__currentIsolatedIsotopicPattern=df.copy().reset_index(drop=True)
                return True


                

        self.__currentIsolatedIsotopicPattern=pd.DataFrame()
        self.__currentIsotopicPatternSection=pd.DataFrame()
        return False            
             
    def __assesIsotopicPattern(self,mzErr=None):
        
        if mzErr!=None: self.isotopicSignals_mzErr=mzErr  
        self.patternsIntensityMtx=np.array([self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.intensity.to_numpy()])
        self.patternsMZMtx=np.array([self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.mz.to_numpy()])
        self.isoPatternToAssess=self.__currentIsolatedIsotopicPattern[['mz','intensity']]
        self.assessIsoPatternByIsoPatternsProjection(evalCandidateFormulas=False)
        return self.patternsVecProjectionErrors
       

    def __filterInteractionProduct(self):
        self.__currentXICPeakAssessment=pd.DataFrame()
        self.currentFSXICPeak=pd.DataFrame()
        self.currentXICPeak=pd.DataFrame()
        if(self.__currentCompoundIdx==-1): self.interactionProductIndexes=(0,0)

        if self.__msCompoundsManager.interactionFormulas.charge > 0:
            FSFilter=self.__FSFilters[self.__FSFilters['polarity']=="+"]
            self.polarity="+"
        elif self.__msCompoundsManager.interactionFormulas.charge < 0:
            FSFilter=self.__FSFilters[self.__FSFilters['polarity']=="-"]
            self.polarity="-"
        else:
            print('W.(msTargetAnalysis): the interaction product is neutral')
            return

        if FSFilter.empty: return
        if self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.empty: return
        __mzTarg=self.__msCompoundsManager.interactionFormulas.exactMass.mz.iloc[0]
        self.setAdqParameters(FSFilter['filter'].iloc[0],FSFilter['polarity'].iloc[0],self.mzAcc,__mzTarg)
        self.chromPeakMinNumOfPoints=self.chromPeakMinNumOfPointsFS
        self.__searchAnalyticalSignal()
        self.currentFSXICPeak=self.currentXICPeak.copy()
        return

        
    def __searchAnalyticalSignal(self):
        self.__currentXICPeakAssessment=pd.DataFrame()
        self.__currentIsolatedIsotopicPattern=pd.DataFrame()
        self.__currentIsotopicPatternSection=pd.DataFrame()
        self.currentXICPeak=pd.DataFrame()

        if self.eventsDF.empty:
            self.events=self.MSfile

        __isCurrtRSet=not(np.isnan(self.__targetCompoundProperties.iloc[[self.__currentCompoundIdx]]['tR'].iloc[0]))
        xicHog=self.getXICArray() 
        if isinstance(xicHog,tuple):

            self.__currentXICPeakAssessment=msMLModelsComm.classifyChromatographicSignal(xicHog[0])

            if __isCurrtRSet:
                __tRprop=self.__targetCompoundProperties.iloc[[self.__currentCompoundIdx]]['tR'].iloc[0]
                multipeaks=self.getMultiPeaksInfo(isPlotShown=False,isPlotSaved=False,xicPeak=xicHog[1])
                __idx=abs(multipeaks.time-__tRprop).idxmin()
                tRmax=multipeaks.time.iloc[__idx]
                __intTotRMax=multipeaks.intensity.iloc[__idx]

            else:
                tRmax=xicHog[1].time.iloc[xicHog[1].intensity.idxmax()]
                __intTotRMax=xicHog[1].intensity.max()

            self.tR=tRmax           
            self.currMS1ScanNum2tRmax=self.eventsDF[self.eventsDF.tR==tRmax].nScan.iloc[0]
            if not(self.__labelIsotopicPattern(self.currMS1ScanNum2tRmax)):
                self.__currentXICPeakAssessment=pd.DataFrame()
                return
            self.__numOfdetectedCompounds+=1
            self.currentXICPeak=xicHog[1].copy()
            self.mzTargetFS=self.mzTarget
            vectorialProj=self.__assesIsotopicPattern()
            self.__currentDetectedCompound=pd.DataFrame({'CompNum':[self.__numOfdetectedCompounds]})
            self.__currentDetectedCompound['ionNum']=self.__numOfdetectedCompounds
            self.__currentDetectedCompound['rankSource']=1
            self.__currentDetectedCompound['isoPatternIdx']=np.nan
            self.__currentDetectedCompound['scanNum2tRmax']=self.currMS1ScanNum2tRmax 
            self.__currentDetectedCompound['IDkey']=f"{self.__currentCompoundIdx}:{self.__currentInteractionProductIdx}"
            self.__currentDetectedCompound['name']=self.__msCompoundsManager.interactionFormulas.name
            self.__currentDetectedCompound['canonicalSmiles']=self.__msCompoundsManager.interactionFormulas.smiles
            self.__currentDetectedCompound['inchikey']=self.__msCompoundsManager.interactionFormulas.inchikey
            self.__currentDetectedCompound['molecularFormula']=self.__msCompoundsManager.interactionFormulas.baseFormula
            self.__currentDetectedCompound['bmonoisotopicMass']=self.__msCompoundsManager.interactionFormulas.baseExactMass.exactMW.iloc[0]
            self.__currentDetectedCompound['polarity']=self.polarity
            self.__currentDetectedCompound['iMax2tRmax']=__intTotRMax
            self.__currentDetectedCompound['iMaxNoise']=self.__currentIsolatedIsotopicPattern.noise.iloc[0]
            self.__currentDetectedCompound['iMaxCharge']=self.__currentIsolatedIsotopicPattern.charge.iloc[0]
            self.__currentDetectedCompound['identificationApproach']=self.__analyticalApproachInferences()
            self.__currentDetectedCompound['interactionProductAccurateMass']=self.accurateMass
            self.__currentDetectedCompound['massError_ppm']=self.relativeMassError
            self.__currentDetectedCompound['ionizedSpecies']=self.__msCompoundsManager.interactionFormulas.formulaTransformation
            self.__currentDetectedCompound['interactionProductCharge']=self.__msCompoundsManager.interactionFormulas.charge
            self.__currentDetectedCompound['interactionProductFormula']=self.__msCompoundsManager.interactionFormulas.ionicFormula
            self.__currentDetectedCompound['einteractionProductFormula']= self.__msCompoundsManager.interactionFormulas.formula
            self.__currentDetectedCompound['interactionProductExactMass']=self.__msCompoundsManager.interactionFormulas.exactMass.exactMW.iloc[0]
            self.__currentDetectedCompound['tRmax']=tRmax
            self.__currentDetectedCompound['%B']=f"{self.getLCGradientOrganicComposition(tRmax):.1F}"
            self.__currentDetectedCompound=self.__currentDetectedCompound.join(vectorialProj[['vectorialProj_intensityError','vectorialProj_mzError','vectorialProj_combinedError']])
            self.__currentDetectedCompound=self.__currentDetectedCompound.join(self.__currentXICPeakAssessment)
            self.__currentDetectedCompound['p_vectorialProj']=vectorialProj.iloc[0]['p_vectorialProj']
            self.__currentDetectedCompound['p_amenability']=self.__pCurrentCompound_Amenbility
            self.__currentDetectedCompound['tR_inference']=self.__currentCompound_tR['tR'].iloc[0]
            self.__currentDetectedCompound['U_tR_inference']=self.__currentCompound_tR['U_tR'].iloc[0]
            self.__currentDetectedCompound['p_tR_inference']=self.__currentCompound_tR['p_tR'].iloc[0]
            self.__currentDetectedCompound['p_ionizedSpecies']=self.__msCompoundsManager.currentIonizedSpeciesProb[self.__currentInteractionProductIdx]
            self.__currentDetectedCompound['p_ms2']=np.nan
            self.__currentDetectedCompound['numMS2Fragments']=0
            self.__currentDetectedCompound['p_ms2Qlty']=np.nan
        return
            
   

    def __ms2SignalsDetection(self,sourceMSn,areIons_tRFiltered=True,tR_thres=0.1):

        ## MSn spectrum data retrieve
        sourceMSn_=sourceMSn

        if isinstance(sourceMSn_,pd.core.frame.DataFrame):
            sourceMSn_=sourceMSn_.copy().sort_values(['mz'],ascending=False)
            sourceMSn_=sourceMSn_[sourceMSn_.mz>50.0].reset_index(drop=True)

        elif isinstance(sourceMSn_,list):
            sourceMSn_=pd.DataFrame({'mz':sourceMSn_}).sort_values(['mz'],ascending=False)
            sourceMSn_=sourceMSn_[sourceMSn_.mz>50.0].reset_index(drop=True)
            
        elif isinstance(sourceMSn_,str):
            sourceMSn_=pd.DataFrame({'mz':sourceMSn_.replace(",",";").split(";")}).astype(float)
            sourceMSn_=sourceMSn_.copy().sort_values(['mz'],ascending=False)
            sourceMSn_=sourceMSn_[sourceMSn_.mz>50.0].reset_index(drop=True)

        else:
            sourceMSn_=None


        self._lcmsMeasurements__setAdqFilters(self.accurateMass,mzAcc=self.__mzAcc_ppm[0],
                                              polarity=self.polarity,
                                              tR_thres=tR_thres)



        
        ionDetectionData=self.searchIon()
        if isinstance(ionDetectionData,tuple):
            ms2_tR_Lims=[ionDetectionData[3]-self.tRError*10.0,ionDetectionData[3]+self.tRError*10.0]
            if areIons_tRFiltered:
                ms2_maxtRErr=[ionDetectionData[3]-self.tRError,ionDetectionData[3]+self.tRError]
            ionDetectionData[1]['MSn']="ms1"
            ionDetectionData[1]['mz']=self.accurateMass
            ionDetectionData[1]['CompNum']=self.__currentCompoundIdx
            self.__ms1XIC=ionDetectionData[1].copy()
            self.setXICFilterMask(self.__ms1XIC)
            if (os.path.exists(self.__targetXICsFile)):
                self.__ms1XIC.to_csv(self.__targetXICsFile,sep="\t",index=False, mode='a',header=False)
            else:
                self.__ms1XIC.to_csv(self.__targetXICsFile,sep="\t",index=False)

                
            self.__signalsDetection(sourceMSn_,
                                    ms2_tR_Lims,ms2_maxtRErr,
                                    ms1Scan=ionDetectionData[4],
                                    filterMask=True,
                                    tR_thres=tR_thres)
        return

    
    def __signalsDetection(self,sourceMSn,ms2_tR_Lims,ms2_maxtRErr,ms1Scan=None,filterMask=False,tR_thres=0.1):
        
        if len(re.findall("[+-]",self.__currPolarity))==0: return


        self.searchTargetProbableFragmentIons(self.accurateMass,
                                              self.__currPolarity,ms2_tR_Lims,ms2_maxtRErr,
                                              self.__mzAcc_ppm[0],sourceMSn,
                                              ms1Scan,filterMask,tR_thres)
              
        self.__msCompoundsManager.interactionFormulas.msFragmenter.msAnnotationData=pd.DataFrame()
        if not(self.__isTargetKnowns): self.__fragmentsIonsAssessment()
        self.__saveDectectionInfo()

        return
        
    def getMZSpectrumPrediction(self):
        self.__msCompoundsManager.interactionFormulas.predictMSfragmentation(saveMSPrediction=True)
        return self.__msCompoundsManager.interactionFormulas.massSpectrumPrediction


    def showCurrentCompoundStructure(self):
        return self.__msCompoundsManager.interactionFormulas.showMolecularStructure()
    
    def plotMZSpectrumPrediction(self,CE="10eV"):
        self.__msCompoundsManager.interactionFormulas.plotMassSpectrumPrediction(CE)

    def plotCurrentXIC(self):

        if (not(self.currentXICPeak.empty)) & (not(self.__currentDetectedCompound.empty)):

            plt.rcdefaults()
            plt.clf()
            plt.plot(self.currentXICPeak.time,self.currentXICPeak.intensity)
            plt.xlabel('time / min')
            plt.ylabel('Intensity')
            plt.title(f"{self.__currentDetectedCompound['name'].iloc[0]} - {self.__currentDetectedCompound.einteractionProductFormula.iloc[0]}\nAccurate m/z: {round(self.accurateMass,5)} Th - Exact m/z: {round(self.mzTarget,5)} Th\n (Mass error: {round(self.relativeMassError,2)} ppm)")
            plt.tight_layout()
            plt.show()


    def plotCurrentIsoPatternSection(self,addTheoretical=False):


        if ( (not(self.__currentIsotopicPatternSection.empty)) & (not(self.__currentIsolatedIsotopicPattern.empty)) ):
            sSignals=self.__currentIsotopicPatternSection[self.__currentIsotopicPatternSection.ionLabel=="S"]
            
            plt.rcdefaults()
            plt.clf()
            if addTheoretical:
                plt.plot(self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.mz,
                         self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.intensity*self.__currentIsolatedIsotopicPattern.intensity.max(),
                         'o',color='orange',markersize=5,label=f"Theoretical pattern",alpha=0.3)

            
            plt.plot(sSignals.mz,sSignals.intensity,'bo',alpha=0.3)
            plt.plot(self.__currentIsolatedIsotopicPattern.mz,self.__currentIsolatedIsotopicPattern.intensity,'go',alpha=0.3,label="Experimental pattern")

            
            
            plt.stem(self.__currentIsolatedIsotopicPattern.mz,self.__currentIsolatedIsotopicPattern.intensity,linefmt='g-',markerfmt='none',basefmt="none")
            plt.ylim(0.0,self.__currentIsotopicPatternSection.intensity.max()*1.05)
            plt.xlabel('m/z Th')
            plt.ylabel('Intensity')
            
            plt.title(f"{self.__currentDetectedCompound['name'].iloc[0]} - {self.__currentDetectedCompound.einteractionProductFormula.iloc[0]}\nAccurate m/z: {round(self.accurateMass,5)} Th - Exact m/z: {round(self.mzTarget,5)} Th\n (Mass error: {round(self.relativeMassError,2)} ppm)",fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.show()            

            
    def plotCurrentIsoPattern(self,addTheoretical=True):

        if self.__currentXICPeakAssessment.empty: return
        if not(self.__currentIsolatedIsotopicPattern.empty):
           
            plt.rcdefaults()
            plt.clf()

            if addTheoretical:
            
                plt.plot(self.__currentIsolatedIsotopicPattern.mz,
                         self.__currentIsolatedIsotopicPattern.intensity/self.__currentIsolatedIsotopicPattern.intensity.max(),
                         'bo',alpha=0.3)
                plt.plot(self.__currentIsolatedIsotopicPattern.mz,
                         self.__currentIsolatedIsotopicPattern.intensity/self.__currentIsolatedIsotopicPattern.intensity.max(),
                         'b',alpha=0.3,label="Experimental pattern")
                plt.plot(self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.mz,
                         self.__msCompoundsManager.interactionFormulas.theoreticalIsotopicPattern.intensity,
                         'o-',color='orange',markersize=5,label=f"Theoretical pattern")
                plt.title(f" Experimental vs. theoretical isotopic pattern\n{self.__currentDetectedCompound['name'].iloc[0]} - {self.__currentDetectedCompound.einteractionProductFormula.iloc[0]}\nAccurate m/z: {round(self.accurateMass,5)} Th - Exact m/z: {round(self.mzTarget,5)} Th\n (Mass error: {round(self.relativeMassError,2)} ppm)",fontsize=11)
                markerline, stemlines, baseline=plt.stem(self.__currentIsolatedIsotopicPattern.mz,self.__currentIsolatedIsotopicPattern.intensity/self.__currentIsolatedIsotopicPattern.intensity.max(),linefmt='gray',markerfmt='none',basefmt="none")
                stemlines.set_alpha(0.2)


                plt.ylim(0.0,1.05)
                plt.legend()
            else:
                plt.stem('mz','intensity',data=self.__currentIsolatedIsotopicPattern,linefmt='g-',markerfmt='none',basefmt="none")                               
                plt.title(f" Experimental isotopic pattern\n{self.__currentDetectedCompound['name'].iloc[0]} - {self.__currentDetectedCompound.einteractionProductFormula.iloc[0]}\nAccurate m/z: {round(self.accurateMass,5)} Th - Exact m/z: {round(self.mzTarget,5)} Th\n (Mass error: {round(self.relativeMassError,2)} ppm)",fontsize=11)
                plt.ylim(0.0,self.__currentIsolatedIsotopicPattern.intensity.max()*1.05)

                
            plt.xlabel('m/z Th')
            plt.ylabel('Intensity')
            plt.tight_layout()
            plt.show()

            
    def plotCurrentMassSpectrum(self):

        if  not(self.__currentIsolatedIsotopicPattern.empty):
        
            self.plotMassSpectrum(title=f"{self.__currentDetectedCompound['name'].iloc[0]} - {self.__currentDetectedCompound.einteractionProductFormula.iloc[0]}",
                                  appendMSdata=self.__currentIsolatedIsotopicPattern)


    def plotCurrentCompoundXIC(self):

        
        if not(self.__currentDetectedCompound.empty):
            self.xic=self.currentFSXICPeak
            dtR=self.__currentDetectedCompound.tR_inference.iloc[0]-self.tR
            self.plotXIC(plot2dtR=dtR,U_tR=self.__currentDetectedCompound.U_tR_inference.iloc[0],title=self.__currentDetectedCompound['name'].iloc[0])
            self.xic=pd.DataFrame()

            
    def plotIonsXICs4CurrentCompound(self,detectionData,isStacked=False):

        if not(detectionData[1].empty):
            gxics=detectionData[1].groupby(['MSn','mz'])
            tRMax=detectionData[0][detectionData[0].MSn=="ms1"].tRmax.iloc[0]
            plt.clf()
            plt.rcdefaults()


            if isStacked & (len(gxics.groups) > 1):
                colors = plt.rcParams["axes.prop_cycle"]()
                fig = plt.figure()
                gs = fig.add_gridspec(len(gxics.groups), hspace=0)
                axs = gs.subplots(sharex=True)

                for i,idata in enumerate(gxics):
                    c = next(colors)["color"]
                    axs[i].plot(idata[1].time,idata[1].intensity,
                             linewidth=0.85,
                             alpha=1.0,
                            color=c,
                             label=f"{idata[0][0]}: {idata[0][1]:.5F}")

                    axs[i].legend(fontsize=7.5)
                    axs[i].axvline(x=tRMax,ymin=0.0,ymax=1.0,color='gray',
                                 linestyle = ':', alpha = 0.5,linewidth=0.85)
                    axs[i].axvline(x=tRMax+self.tRError,ymin=0.0,ymax=1.0,color='gray',
                                 linestyle = ':', alpha = 0.5,linewidth=0.85)
                    axs[i].axvline(x=tRMax-self.tRError,ymin=0.0,ymax=1.0,color='gray',
                                 linestyle = ':', alpha = 0.5,linewidth=0.85)
                    
                    axs[i].set_xlim(tRMax-1.0,tRMax+1.0)
                    axs[i].tick_params(axis='y',labelsize = 8, width=2)
                    axs[i].ticklabel_format(axis='y',useMathText=True,
                                            scilimits=(0,0),useOffset=False,style='sci')

                fig.supxlabel('time / min')
                fig.supylabel('Intensity')
                plt.show()

            elif len(gxics.groups) > 1:
                for idata in gxics:
                    plt.plot(idata[1].time,idata[1].intensity,
                             linewidth=0.85,
                             alpha=1.0,
                             label=f"{idata[0][0]}: {idata[0][1]:.5F}")
                    plt.ticklabel_format(axis='y',useMathText=True,
                                            scilimits=(0,0),useOffset=False,style='sci')


                plt.xlim(tRMax-1.0,tRMax+1.0)
                plt.xlabel('time / min')
                plt.ylabel('Intensity')
                plt.axvline(x=tRMax,ymin=0.0,ymax=1.0,color='gray',
                            linestyle = ':', alpha = 0.5,linewidth=0.85)
                plt.axvline(x=tRMax+self.tRError,ymin=0.0,ymax=1.0,color='gray',
                               linestyle = ':', alpha = 0.5,linewidth=0.85)
                plt.axvline(x=tRMax-self.tRError,ymin=0.0,ymax=1.0,color='gray',
                               linestyle = ':', alpha = 0.5,linewidth=0.85)
                plt.legend(fontsize=7.5)
                plt.show()

    def searchSingleTargetCompound(self,areIons_tRFiltered=True,forceFragmentation=False,mzPrecursor=None,tR_thres=0.1):

        self.__currentDetectedCompound=pd.DataFrame()
        if self.__targetCompoundProperties.empty: return

        self.__numOfIsoSignalsFilter=self.numOfIsoSignals       
        if not(self.__targetCompoundProperties.iloc[self.__currentCompoundIdx].isna().any()):
            self.__numOfIsoSignalsFilter=1
            self.__isTargetKnowns=True
        else:
            self.__isTargetKnowns=False

       
        tmpIntThres=self.__intensityThresError
        if ( (self.__targetCompoundProperties.name.iloc[0].find("(IS)")>0) | (self.__targetCompoundProperties.name.iloc[0].find("(SS)")>0)):
            self.__intensityThresError=list(np.array(self.__intensityThresError)*2.0)

        self.__filterInteractionProduct() 
        self.__intensityThresError=tmpIntThres   
        if self.__currentXICPeakAssessment.empty: return ()
        __filteringFeatures=self.__currentDetectedCompound.iloc[0]
        self.__detected+=1


        if ((__filteringFeatures['qltySignal']==True) & \
            (__filteringFeatures['p_vectorialProj']>=self.vectorialProjProbThreshold) & \
            (__filteringFeatures['p_tR_inference']>=self.__tRProbThreshold)):
        
    
            sourceMSn=self.__targetCompoundProperties.iloc[self.__currentCompoundIdx].fragmentationProfile
            if str(sourceMSn)=="nan":
                sourceMSn=None
            elif isinstance(sourceMSn,float):
                sourceMSn=[sourceMSn]

            self.__ms2SignalsDetection(sourceMSn,areIons_tRFiltered,tR_thres=tR_thres)

            __isoPatt=self.currentIsolatedIsotopicPattern.copy()
            __isoPatt['CompNum']=self.__numOfdetectedCompounds
            if (os.path.exists(self.__targetIsotopicPatternsFile)):
                __isoPatt.to_csv(self.__targetIsotopicPatternsFile,sep="\t",index=False, mode='a',header=False)
            else:
                __isoPatt.to_csv(self.__targetIsotopicPatternsFile,sep="\t",index=False)

        self.__allDetectedCompounds=pd.concat([self.__allDetectedCompounds,self.__currentDetectedCompound],sort=False,ignore_index=True)
        self.__allDetectedCompounds.to_csv(self.__ionsDetectionFile,sep="\t",index=False)
        return
                

    def searchTargetCompounds(self,forceSearch=False):
        self.__currentInteractionProductIdx=-1
        self.__currentCompoundIdx=-1
        self.__numOfdetectedCompounds=0
        self.__identProcessingtime=pd.DataFrame()
        self.__allDetectedCompounds=pd.DataFrame()
        self.__detected=0

        if not(os.path.exists(self.__resultsPath)):
            os.mkdir(self.__resultsPath)
            
        
        if not(forceSearch):
            if os.path.exists(self.__ionsDetectionFile): return

        if os.path.exists(self.__targetXICsFile): os.remove(self.__targetXICsFile)
        if os.path.exists(self.__ionsDetectionFile): os.remove(self.__ionsDetectionFile)
        if os.path.exists(self.__targetFragmentsXICsFile): os.remove(self.__targetFragmentsXICsFile)
        if os.path.exists(self.__targetFragmentsIonsFoundFile): os.remove(self.__targetFragmentsIonsFoundFile)
        if os.path.exists(self.__targetFragmentsIonsFoundAnnotatedFile): os.remove(self.__targetFragmentsIonsFoundAnnotatedFile)
        if os.path.exists(self.__targetIsotopicPatternsFile): os.remove(self.__targetIsotopicPatternsFile)
        if os.path.exists(self.__elapsedTimeFile): os.remove(self.__elapsedTimeFile)

        if self.rawFile != None:
            self.__numOfdetectedCompounds=0
            gstartTime = time.monotonic()
            compoundFSDetectionDF=pd.DataFrame()
            compoundAndFragmentsDetectionDF=pd.DataFrame()
            pbarSpects = tqdm(total=self.totalSignalToSearch, bar_format='{l_bar}{bar:100}{r_bar}{bar:-5b}')
            
            for count,compound in enumerate(self.__iter__()):

                startTime = time.monotonic()
                os.system('clear')
                clear_output(wait=False)                
                pbarSpects.set_description(f"Targeted processing (njobs: {self.njobs}): compound {self.currentTargetCompound.inchikey} - {os.path.basename(self.rawFile)}  - Detected: {self.__detected}")    

                self.searchSingleTargetCompound()
                elapsed_time = time.monotonic() - startTime
                self.__identProcessingtime=pd.concat([self.__identProcessingtime,
                                                    pd.DataFrame({"ionNum":[count+1],
                                                                  "processTime":[elapsed_time]
                                                                  })],sort=False,ignore_index=True)
                self.__identProcessingtime.to_csv(self.__elapsedTimeFile,sep="\t",index=False)
                pbarSpects.update(1)


            clear_output(wait=True)
            pbarSpects.bar_format       
            pbarSpects.close()


            elapsed_time = time.monotonic() - gstartTime
            print(f"Global target search time: {round(elapsed_time,3)} s")
            self.__identProcessingtime['totalTime']=elapsed_time
            self.__identProcessingtime.to_csv(self.__elapsedTimeFile,sep="\t",index=False)
            self.__allDetectedCompounds.to_csv(self.__ionsDetectionFile,sep="\t",index=False)


