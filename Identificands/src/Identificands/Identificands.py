import sys,os,glob
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from decimal import Decimal

from IPython.display import display, clear_output
from tqdm import tqdm


sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from msNontargetedAnalysis import msNontargetedAnalysis
from msTargetAnalysis import msTargetAnalysis
from msIdentificands import  msIdentificands
from msCompoundExplorer import msCompoundExplorer



class Identificands(msIdentificands):
    
    def __init__(self):
        msIdentificands.__init__(self)
        self.__baseRawFileName=""
        self.compoundExplorer=msCompoundExplorer()

        #Framework analysis
        self.__identificandsReferences=None
        self.__targetInputReferences=None
        self.__analysisFrameworks={'targeted-analysis':False,'suspect-screening':False,'non-targeted-screening':False}
        self.nontargetedAnalysis=None
        self.targetAnalysis=None
        self.__njobs=8
        self.__nontargetedASAssesmentMinNPoints=2
        self.__filesToProcess=pd.DataFrame()
        self.__eidentificandsReferences=pd.DataFrame({"inchikey":[None],"name":[None],"smiles":["X"],"ionizedSpecies":None,"fragmentationProfile":None,"tR":None,"amenability":None})
        self.__fileToProcesIdx=0
        self.__currentRawFile=""
        self.__intensityThreshold=self.compoundExplorer.intensityThreshold

    @property
    def filesToProcess(self):
        return self.__filesToProcess

    @filesToProcess.setter
    def filesToProcess(self,processFile):
        if os.path.exists(processFile):
            self.__filesToProcess=pd.read_csv(processFile,sep="\t")

            if "analyze" in self.__filesToProcess.columns:
                self.__filesToProcess=self.__filesToProcess[self.__filesToProcess['analyze']!=False]
                self.__filesToProcess=self.__filesToProcess.drop(columns=['analyze']).reset_index(drop=True)

                if self.__filesToProcess.empty:
                    self.__analysisFrameworks={'targeted-analysis':False,'suspect-screening':False,'non-targeted-screening':False}
                    print(f"W.(Identificands): {processFile}. Nothing file to process")
                    return

            self.fileToProcesIdx=0

        else:
            print(f"W.(Identificands): {processFile} file not found")


    @property
    def fileToProcesIdx(self):
        return self.__fileToProcesIdx

    @fileToProcesIdx.setter
    def fileToProcesIdx(self,value):
        self.__fileToProcesIdx=value
        __idx=self.__fileToProcesIdx%len(self.__filesToProcess)
        self.__currentRawFile=self.__filesToProcess.iloc[__idx].fileToProcess
        if os.path.exists(self.__currentRawFile):
            __identificandsReferences=self.__filesToProcess.iloc[__idx].inclusionList
           
            if str(__identificandsReferences)=="nan": __identificandsReferences=self.__eidentificandsReferences
            self.identificandsReferences=__identificandsReferences
            self.rawFile=self.__currentRawFile
        else:
            print(f"W.(Identificands): {self.__currentRawFile} file not found")
        
        
    @property
    def targetInputReferences(self):
        return self.__targetInputReferences


    @property
    def lcmsTools(self):
        if isinstance(self.nontargetedAnalysis,type(None)):
            self.nontargetedAnalysis=msNontargetedAnalysis()
            self.nontargetedAnalysis.intensityThreshold=self.__intensityThreshold
            self.nontargetedAnalysis.njobs=self.njobs
            self.nontargetedAnalysis.asAssesmentMinNPoints=self.__nontargetedASAssesmentMinNPoints
            self.nontargetedAnalysis.rawFile=self.rawFile

        return self.nontargetedAnalysis

        
    @property
    def njobs(self):
        return self.__njobs

    @njobs.setter
    def njobs(self,value):
        self.__njobs=value


    @property
    def intensityThreshold(self):
        return self.__intensityThreshold
        
    @intensityThreshold.setter
    def intensityThreshold(self,value):
        self.__intensityThreshold=value
        if not(isinstance(self.nontargetedAnalysis,type(None))):
            self.nontargetedAnalysis.intensityThreshold=self.__intensityThreshold
        if not(isinstance(self.targetAnalysis,type(None))):
            self.targetAnalysis.intensityThreshold=self.__intensityThreshold
    
    @property
    def analysisFrameworks(self):

        aF=pd.DataFrame.from_dict([self.__analysisFrameworks]).T.rename(columns={0:'Analysis frameworks'})
        if not(aF['Analysis frameworks'].any()):
            if not(self.identificandsData.empty):
                self.__analysisFrameworks={key: True for key in self.__analysisFrameworks}
                for idApp in  self.identificandsData.identificationApproach.drop_duplicates().str.replace("unknowns","non-targeted-screening").to_list():
                    self.__analysisFrameworks[idApp]=True
                    aF=pd.DataFrame.from_dict([self.__analysisFrameworks]).T.rename(columns={0:'Analysis frameworks'})
        return  aF

               
    @property
    def identificandsReferences(self):
        return self.__identificandsReferences


    @identificandsReferences.setter
    def identificandsReferences(self,value):
        if isinstance(value,str):
            if os.path.exists(value):
                self.__identificandsReferences=pd.read_csv(value,sep="\t")
            else:
                return
        else:
            self.__identificandsReferences=value

        if "analyze" in self.__identificandsReferences.columns:
            self.__identificandsReferences=self.__identificandsReferences[self.__identificandsReferences['analyze']!=False]
            self.__identificandsReferences=self.__identificandsReferences.drop(columns=['analyze']).reset_index(drop=True)
           
        self.__analysisFrameworks={'targeted-analysis':False,'suspect-screening':False,'non-targeted-screening':False}
        self.__targetInputReferences=self.__identificandsReferences.copy().drop_duplicates()
        __frameworkAssess=self.__targetInputReferences[(self.__targetInputReferences.smiles=="x") | (self.__targetInputReferences.smiles=="X")]
        if not(__frameworkAssess.empty):
            self.__targetInputReferences=self.__targetInputReferences.drop(__frameworkAssess.index.to_list(),axis=0)
            self.__analysisFrameworks['non-targeted-screening']=True
        del __frameworkAssess

        if self.__targetInputReferences.isna().T.apply('any').any():
            self.__analysisFrameworks['suspect-screening']=True

        if not(self.__targetInputReferences.isna().T.apply('any').all()):
            self.__analysisFrameworks['targeted-analysis']=True
            
    def startNontargetedAnalysis(self,forceSearch=False):
        if self.__analysisFrameworks['non-targeted-screening']:
            print(f"Starting non-targeted screening for {os.path.basename(self.rawFile)}")
            self.nontargetedAnalysis=msNontargetedAnalysis()
            self.nontargetedAnalysis.intensityThreshold=self.__intensityThreshold
            self.nontargetedAnalysis.resultsPath="/".join(self.resultsPath.split("/")[0:-1])
            self.nontargetedAnalysis.njobs=self.njobs
            self.nontargetedAnalysis.asAssesmentMinNPoints=self.__nontargetedASAssesmentMinNPoints
            self.nontargetedAnalysis.rawFile=self.rawFile
            self.nontargetedAnalysis.assessCandidateAnalyticalSignals(forceSearch=forceSearch)
            self.nontargetedAnalysis.searchNontargetedCompounds(forceSearch=forceSearch)



    def startTargetAnalysis(self,forceSearch=False):
        if self.__analysisFrameworks['suspect-screening'] | self.__analysisFrameworks['targeted-analysis']:

            __inRefs=self.targetInputReferences.copy()
            if (self.__analysisFrameworks['suspect-screening'] & (not(self.__analysisFrameworks['targeted-analysis'])) ):
                __mode="suspect-screening"
                __modeN="suspectScreening"
            elif (self.__analysisFrameworks['targeted-analysis'] & (not(self.__analysisFrameworks['suspect-screening'])) ):
                __mode="targeted-analysis"
                __modeN="targetedAnalysis"
            else:
                __mode="targeted-screening"
                __modeN="targetedScreening"
                if ( (not(self.existTargetKnownsDetectionDataFile)) & (self.existTargetSuspectsDetectionDataFile) ):
                    __inRefs=self.targetInputReferences.copy().dropna().reset_index(drop=True)
                    __mode="targeted-analysis"
                    __modeN="targetedAnalysis"
                elif ( (self.existTargetKnownsDetectionDataFile) & (not(self.existTargetSuspectsDetectionDataFile)) ):
                    __inRefs=__inRefs.drop(__inRefs.dropna().index,axis=0).reset_index(drop=True)
                    __mode="suspect-screening"
                    __modeN="suspectScreening"

            print(f"Starting {__mode} for {os.path.basename(self.rawFile)}")
            self.targetAnalysis=msTargetAnalysis()
            self.targetAnalysis.resultsPath="/".join(self.resultsPath.split("/")[0:-1])
            self.targetAnalysis.njobs=self.njobs
            self.targetAnalysis.targetMode=__modeN
            self.targetAnalysis.targetCompounds=__inRefs
            self.targetAnalysis.rawFile=self.rawFile
            self.targetAnalysis.searchTargetCompounds(forceSearch=forceSearch)

            
    def startIdentification(self,forceSearch=False):
        self.setAssesmentData()

        __t=(self.identificandsData.empty) |  ( (self.__analysisFrameworks['targeted-analysis'] & self.__analysisFrameworks['suspect-screening'] ) & ( (not(self.existTargetDetectionDataFile)) & ( (not(self.existTargetKnownsDetectionDataFile))  | (not(self.existTargetSuspectsDetectionDataFile)) ) ) )
        __tn=(self.identificandsData.empty) |  ( (self.__analysisFrameworks['targeted-analysis'] & (not(self.__analysisFrameworks['suspect-screening'])) ) & (not(self.existTargetKnownsDetectionDataFile)))
        __ts=(self.identificandsData.empty) |  ( (self.__analysisFrameworks['suspect-screening'] & (not(self.__analysisFrameworks['targeted-analysis'])) ) & (not(self.existTargetSuspectsDetectionDataFile)))

        if (__t | __tn | __ts):
            self.startTargetAnalysis(forceSearch=forceSearch)
            if (self.__analysisFrameworks['targeted-analysis'] | self.__analysisFrameworks['suspect-screening'] ):
                del  self.targetAnalysis
            self.setAssesmentData()


        if ((self.identificandsData.empty) |  ( self.__analysisFrameworks['non-targeted-screening'] & (not(self.existNontargetedDetectionDataFile)) )):

            self.startNontargetedAnalysis(forceSearch=forceSearch)
            if self.__analysisFrameworks['non-targeted-screening']:
                del  self.nontargetedAnalysis
            self.setAssesmentData()


        return


    def identifyMultipleFiles(self):

        if not(self.filesToProcess.empty):

            pbarSpects = tqdm(total=len(self.__filesToProcess), bar_format='{l_bar}{bar:100}{r_bar}{bar:-5b}')
            
            for __file in self.__filesToProcess.iterrows():
                clear_output(wait=False)
                os.system('clear')
                self.__currentRawFile=__file[1].fileToProcess               
                if os.path.exists(self.__currentRawFile):
                    pbarSpects.set_description(f"Processing file: {os.path.basename(self.__currentRawFile)}")
                    __identificandsReferences=__file[1].inclusionList
                    if str(__identificandsReferences)=="nan": __identificandsReferences=self.__eidentificandsReferences
                    if "blankPath" in __file[1].index: self.blankPath=str(__file[1].blankPath)
                    self.identificandsReferences=__identificandsReferences
                    self.rawFile=self.__currentRawFile
                    self.startIdentification()
                    pbarSpects.update(1)
            clear_output(wait=True) 
            pbarSpects.bar_format
            pbarSpects.close()

        self.__filesToProcess=pd.DataFrame()


