# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import time
import os
import re
import subprocess
import pandas as pd
from scipy.stats import norm

#pd.options.mode.copy_on_write = True
import numpy as np
from scipy.stats import chi2
from scipy.stats import f
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import itertools
import io
from PIL import Image
from skimage.transform import resize
from skimage.feature import hog
from datetime import  datetime
import multiprocessing
from joblib import Parallel, delayed

sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])


### Parallelization functions
def getMSScan(scanInfo,rawFile,cmd,isCentroid=True):

    nScan=scanInfo.nScan
    tR=scanInfo.tR
    polarity=scanInfo.polarity
    processingType="centroid"
    if( not(isCentroid) ): processingType="profile"  
    cmd_args=" '"+rawFile+"' -g scanData -n "+str(nScan)+" -t "+processingType+" | sed 's:,:;:g' | grep -iv 'Closing' | grep '^[0-9]'"

    try:
        proc=subprocess.Popen(cmd+cmd_args,stdout=subprocess.PIPE,shell=True)
        df = pd.read_csv(proc.stdout,sep=";",header=None)       
    except:
        return pd.DataFrame()

    if not(df.empty):
        if df.shape[1]==7:
            df.columns=['num','mz','intensity','resolution','charge','baseline','noise']
            df['tR']=tR
            df['polarity']=polarity
        elif df.shape[1]==3:
            df.columns=['num','mz','intensity']
            df['tR']=tR
            df['polarity']=polarity
        else:
            return pd.DataFrame()
        return df
    else:
        return pd.DataFrame()


def getChromatogram(mzTarget,MSFile,polarity,MSfilter,mzAcc,cmd,minNPoints=3):
    df = pd.DataFrame()
    cmd_args='"'+MSFile+'" -g chromatogram  '+'-trace massRange -mzFilter "'+polarity.strip()+" "+MSfilter+'" -mzAcc '+str(mzAcc)+' -massRange '+' "['+str(mzTarget*(1.0-mzAcc/1E6))+';'+str(mzTarget*(1.0+mzAcc/1E6))+']" | grep -iv "Closing"' 

    proc=subprocess.Popen(cmd+cmd_args,stdout=subprocess.PIPE,shell=True)

    try:
        df = pd.read_csv(proc.stdout,sep=";",header=None)
        df.columns=['time','intensity']
        if (df[df['intensity'] >= 1E-6]).shape[0]<minNPoints: return pd.DataFrame()
    except:
        return pd.DataFrame()
    return df


def assessMS1_MS2_signals(ms1XIC,ms2XIC,filterMaskLims,chromPeakMinNumOfPointsMS2,thres=1.75,isChi2Metric=False,dInt=0.2,maxReg=0.1):

    ms2XIC_masked=ms2XIC[(ms2XIC.time>=filterMaskLims[0]) &
                         (ms2XIC.time<=filterMaskLims[1])].copy()

    ms2XIC_masked_maxInt=ms2XIC_masked[(ms2XIC_masked.time>=maxReg[0]) &  (ms2XIC_masked.time<=maxReg[1])].intensity.max()
    ms2XIC_masked.intensity=ms2XIC_masked.intensity/(ms2XIC_masked_maxInt+1E-8)
    ms2XIC_masked=ms2XIC_masked[ms2XIC_masked.intensity<=1.0].reset_index(drop=True)

    
    __ms2XICShape=(ms2XIC_masked[ms2XIC_masked['intensity'] >= 1E-6]).shape
    if __ms2XICShape[0]>=chromPeakMinNumOfPointsMS2:
        ms1XICSection=ms1XIC.copy()
        ms1XICSection=ms1XICSection[(ms1XICSection.time>=filterMaskLims[0]) &  (ms1XICSection.time<=filterMaskLims[1])]
        ms1SampledXICSection=ms1XICSection.copy()
             
        nearIdx=[abs(ms1SampledXICSection.time-t).idxmin() for t in ms2XIC_masked.time] 
        ms1SampledXICSection=ms1SampledXICSection.loc[nearIdx]
        ms1SampledXICSection.intensity=ms1SampledXICSection.intensity/ms1SampledXICSection.intensity.max()    

        if isChi2Metric:
            peakSimilarity=np.sum((ms2XIC_masked.intensity.to_numpy()-ms1SampledXICSection.intensity.to_numpy())**2/(ms1SampledXICSection.intensity.to_numpy()+1.0E-6))
            p_=1-chi2.cdf(peakSimilarity,len(ms2XIC_masked.intensity))
        else:
            SS=sum((ms2XIC_masked.intensity.to_numpy()-ms1SampledXICSection.intensity.to_numpy())**2)
            peakSimilarity=SS/(len(ms2XIC_masked.intensity)*(dInt**2))
            p_=1.0-f.cdf(peakSimilarity,__ms2XICShape[0] , 1000, loc=0, scale=1)

        return (ms1SampledXICSection,p_,ms2XIC_masked,ms2XIC_masked_maxInt)
    return (pd.DataFrame(),0.0,pd.DataFrame(),0.0)



def assessFragments(mz,MSfile,polarity,MSfilter,mzAcc,cmd,filterMaskLims,p_thres,xicMS1Template,intensity,chromPeakMinNumOfPointsMS2,isChi2Metric=False,dInt=0.2,maxReg=0.1):
    __probableFragmentIonsXICs=pd.DataFrame()
    __probableFragmentIons=pd.DataFrame()
    ionSearchXIC=getChromatogram(mz,MSfile,polarity,MSfilter,mzAcc,cmd,chromPeakMinNumOfPointsMS2)

    if not(ionSearchXIC.empty):
        peaksComparisonResults=assessMS1_MS2_signals(xicMS1Template,ionSearchXIC,filterMaskLims,chromPeakMinNumOfPointsMS2,isChi2Metric=isChi2Metric,dInt=dInt,maxReg=maxReg)
           
        if peaksComparisonResults[1]>p_thres:
            peaksComparisonResults[0]['mz']=mz
            peaksComparisonResults[0]['label']=f"ms1SampledSection"
            peaksComparisonResults[0]['p']=peaksComparisonResults[1]
            peaksComparisonResults[0]['intMax']=peaksComparisonResults[3]
            __probableFragmentIonsXICs=peaksComparisonResults[0].copy()
            peaksComparisonResults[2]['mz']=mz
            peaksComparisonResults[2]['label']=f"ms2MaskedSection"
            peaksComparisonResults[2]['p']=peaksComparisonResults[1]
            peaksComparisonResults[2]['intMax']=peaksComparisonResults[3]
            __probableFragmentIonsXICs=pd.concat([__probableFragmentIonsXICs,peaksComparisonResults[2]],sort=False,ignore_index=True)
            __probableFragmentIons=pd.DataFrame({"mz":[mz],
                          "intensity":[intensity],
                          "tR_maxInt":[peaksComparisonResults[2].time[peaksComparisonResults[2].intensity.idxmax()]],
                          "maxInt":[peaksComparisonResults[3]],
                          "p_MS1_MS2":[peaksComparisonResults[1]]
                          })
            return (__probableFragmentIons,__probableFragmentIonsXICs)
    return (__probableFragmentIons,__probableFragmentIonsXICs)


class lcmsMeasurements():

    def __init__(self):
        self.__commandPath=os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],"External/LARPRawReader/bin/Release/net6.0/")
        self.__command="LARPRawReader"
        self.__cmd=self.__commandPath+self.__command+" "
        self.__chromPeakMinNumOfPoints=3
        self.__chromPeakMinNumOfPointsFS=3
        self.__chromPeakMinNumOfPointsMS2=2
        self.__numOfIsoSignals=3
        self.__njobs=8
        self.__MSfile=""

        #Ion signals
        self.__xic=pd.DataFrame()
        self.__ms=pd.DataFrame()
        self.__probableFragmentIons=pd.DataFrame()
        self.__currentIonSpeciesList=pd.DataFrame()
        self.__currentIonSpecies=pd.DataFrame()
        self.__completeMS=pd.DataFrame()
        self.__annotatedFragmentIons=pd.DataFrame()
       
        # Parameters and settings for measurement & adquisition signals
        self.__MSfilter=" Full ms "
        self.__ms2Filter="--"
        self.__polarity="+"
        self.__polarityLbl="Positive"
        self.__currentCharge=1.0
        self.__mzAcc=5.0 
        self.__mzTarget=None 
        self.__mzTargetFS=None 
        self.__mzTargetPrecursor=0.0 
        self.__deltaMZ=None
        self.__nearMS2Scans=[]
        self.__nearMS2Events=pd.DataFrame()
        self.__nearestMS2ScanIdx=0
        self.__currMS1ScanNum2tRmax=None
        self.__spectrumDataType=""
        self.__SN_thres=10.0
        self.__retentionTimeError=0.1 #min
        self.__currentCE=None

        #Isotopic pattern section
        self.__accurateMass=0.0 
        self.__rMassError=0.0 
        self.__tR=None
        self.__completeExpIsotopicPattSection=pd.DataFrame()
        self.__isotopicPattUpperBound=None
        self.__isotopicPattLowerBound=None
      
        #Events files
        self.__eventsFile=""
        self.__eventsState=False
        self.__events=pd.DataFrame()

        

        #Adquisition parameters & filters
        self.__adqParameters=pd.DataFrame()
        self.__adqParametersFile=""
        self.__filters=pd.DataFrame()
        self.__filtersFile=""
        self.__filterMaskLims=[]
        self.__isAdqFilterSetted=False
        self.__gradient=pd.DataFrame()
        self.__intensityThres=2E4 

        #Ploting and data related parameters
        self.__resultsPath="./"
        self.__nameSufix=""
        self.__namePrefix=""
        self.__fileName="" 

        ## Chromatografic signals
        self.__currentXICPeak=pd.DataFrame()
        self.__currentFSXICPeak=pd.DataFrame()
        self.__multPeaksInfo=pd.DataFrame()
        self.__probableFragmentIonsXICs=pd.DataFrame()
        

        # Fragmentation profile
        self.fragmentationChi2Metric=False
        self.fragmentationXIC_dInt=0.2
        
        #Debuging info
        self.__isVerbose=False

    @property
    def accurateMass(self):
        return self.__accurateMass

    @accurateMass.setter
    def accurateMass(self,value):
        self.__accurateMass=value

    @property
    def currentCE(self):
        return self.__currentCE

    @currentCE.setter
    def currentCE(self,value):
        self.__currentCE=value

    @property
    def retentionTimeError(self):
        return self.__retentionTimeError

    @retentionTimeError.setter
    def retentionTimeError(self,value):
        self.__retentionTimeError=value
    
    @property
    def relativeMassError(self):
        return self.__rMassError

    @property
    def fileName(self):

        pltName=""
        if self.__fileName!="":
            pltName=self.__nameSufix+self.__fileName+"_"+str(format(self.__mzTarget,'1.2f'))+"_"+self.__polarityLbl+"_"+os.path.basename(self.__MSfile).replace(".raw","")+self.__namePrefix
            
        return pltName

    @fileName.setter
    def fileName(self,value):
        self.__fileName=value

    @property
    def MSfile(self):
        return self.__MSfile

    @MSfile.setter
    def MSfile(self, value):
        self.__MSfile = value
        self.__adqParametersFile=""
        self.__filtersFile=""
        self.__events=pd.DataFrame()
        self.__eventsState = False
        self.__gradient=pd.DataFrame()
        self.__nearMS2Scans=[]
        self.__nearMS2Events=pd.DataFrame()
        self.__nearestMS2ScanIdx=0
        self.__probableFragmentIons=pd.DataFrame()
        self.__currentIonSpecies=pd.DataFrame()
        self.__currentIonSpeciesList=pd.DataFrame()
        self.__ms2Filter="--"
        self.__completeMSFile=os.path.join(self.resultsPath,f"{os.path.basename(self.MSfile).replace('.raw','')}_completeFSMassSpectrum.tsv")


    @property
    def events(self):
        return self.__eventsState


    @property
    def eventsFile(self):
        return self.__eventsFile
    

    @events.setter
    def events(self, file):
        if (self.__eventsState & (self.__eventsFile==file)):
            return
        
        self.__events=self.getEvents(file)
        if self.__events.shape[0]!=0:
            self.__eventsState = True
        else:
            self.__eventsState = False
            self.__eventsFile=""
            
    @property
    def eventsDF(self):
        return self.__events

    @property
    def currentIonSpecies(self):
        return self.__currentIonSpecies

    @currentIonSpecies.setter
    def currentIonSpecies(self,value):
        self.__currentIonSpecies=value

    @property
    def currentIonSpeciesList(self):
        return self.__currentIonSpeciesList

    @currentIonSpeciesList.setter
    def currentIonSpeciesList(self,value):
        self.__currentIonSpeciesList=value.copy() 

        
    @property
    def tR(self):
        return self.__tR
    
    @tR.setter
    def tR(self, value):
        self.__tR=value

    @property
    def currMS1ScanNum2tRmax(self):
        return self.__currMS1ScanNum2tRmax
    
    @currMS1ScanNum2tRmax.setter
    def currMS1ScanNum2tRmax(self, value):
        self.__currMS1ScanNum2tRmax=value
                
    @property
    def gradient(self):
        if self.__gradient.empty:
            self.getElutionInfo()
            
        return self.__gradient

    @property
    def njobs(self):
        return self.__njobs

    @njobs.setter
    def njobs(self,value):
        self.__njobs=value

    
    @property
    def adqExperiments(self):

        if self.__adqParametersFile != self.__MSfile:
            self.__getAdqMode()
        
        return self.__adqParameters
    
    @property
    def intensityThreshold(self):       
        return self.__intensityThres
    
    @intensityThreshold.setter
    def intensityThreshold(self,value):       
        self.__intensityThres=value
    

       
    @property
    def filters(self):

        if self.__filtersFile != self.__MSfile:
            self.__filtersFile=self.__MSfile
            self.__filters=self.getFilters()

        return self.__filters

    @property
    def currentXICPeak(self):
        return(self.__currentXICPeak)

    @currentXICPeak.setter
    def currentXICPeak(self,value):
        self.__currentXICPeak=value 

    @property
    def currentFSXICPeak(self):
        return(self.__currentFSXICPeak)

    @currentFSXICPeak.setter
    def currentFSXICPeak(self,value):
        self.__currentFSXICPeak=value 


    @property
    def probableFragmentIonsXICs(self):
        return self.__probableFragmentIonsXICs

    @probableFragmentIonsXICs.setter
    def probableFragmentIonsXICs(self,value):
        self.__probableFragmentIonsXICs=value

    @property
    def annotatedFragmentIons(self):
        return self.__annotatedFragmentIons

    @annotatedFragmentIons.setter
    def annotatedFragmentIons(self,value):
        self.__annotatedFragmentIons=value
        
    @property
    def multPeaksInfo(self):
        if self.__multPeaksInfo.empty:
            self.getMultiPeaksInfo(isPlotShown=False)
        return self.__multPeaksInfo
        
    @property
    def lastMS2Filter(self):       
        return self.__ms2Filter

    @property
    def nearMS2Scans(self):       
        return self.__nearMS2Scans

    @property
    def nearMS2Events(self):       
        return self.__nearMS2Events

    @property
    def nearestMS2ScanIdx(self):       
        return self.__nearestMS2ScanIdx

    @nearestMS2ScanIdx.setter
    def nearestMS2ScanIdx(self,value):       
        self.__nearestMS2ScanIdx=value
    
    @property
    def filterMaskLims(self):
        return self.__filterMaskLims

    @property
    def isVerbose(self):
        return self.__isVerbose

    @isVerbose.setter
    def isVerbose(self,value):
        self.__isVerbose=value
    
    @property
    def xic(self):
        return self.__xic

    @xic.setter
    def xic(self,value):
        self.__xic=value
    
    @property
    def ms(self):
        return self.__ms


    @ms.setter
    def ms(self,value):
        self.__ms=value

    @property
    def completeMS(self):
        return self.__completeMS

    @completeMS.setter
    def completeMS(self,value):
        self.__completeMS=value
           
    @property
    def isotopicPattUpperBound(self):
        return self.__isotopicPattUpperBound

    @isotopicPattUpperBound.setter
    def isotopicPattUpperBound(self, value):
        self.__isotopicPattUpperBound=value       
    
    @property
    def MSfilter(self):
        return self.__MSfilter

    @MSfilter.setter
    def MSfilter(self, value):
        self.__MSfilter = value    

    @property
    def polarity(self):
        return self.__polarity

    @polarity.setter
    def polarity(self, value):       
        self.__polarityLbl="Positive"
        polty=value
        if( polty=="Positive"):
            self.__polarity="+"
        elif( polty=="Negative"):
            self.__polarity="-"
            self.__polarityLbl="Negative"
        else:
            if polty.strip()=="+":
                self.__polarityLbl="Positive"
                self.__polarity="+"
            elif polty.strip()=="-":
                self.__polarityLbl="Negative"
                self.__polarity="-" 
            else:
                self.__polarity="+"

    @property
    def currentCharge(self):
        return self.__currentCharge

    @currentCharge.setter
    def currentCharge(self,value):
        self.__currentCharge=value
    
    @property
    def deltaMZ(self):
        return self.__deltaMZ

    
    def updateDeltaMZ(self):
        self.__deltaMZ=(self.__mzAcc/1.0E6)*self.__mzTarget
    
    @property
    def mzAcc(self):
        return self.__mzAcc

    @mzAcc.setter
    def mzAcc(self, value):
        self.__mzAcc = value
        self.updateDeltaMZ()

    @property
    def mzTarget(self):
        return self.__mzTarget

    @mzTarget.setter
    def mzTarget(self, value):
        self.__mzTarget = value
        self.__isotopicPattUpperBound=self.__mzTarget+10.0
        self.updateDeltaMZ()


    @property
    def mzTargetPrecursor(self):
        return self.__mzTargetPrecursor

    @mzTargetPrecursor.setter
    def mzTargetPrecursor(self, value):
        self.__mzTargetPrecursor = value

    @property
    def mzTargetFS(self):
        return self.__mzTargetFS

    @mzTargetFS.setter
    def mzTargetFS(self, value):
        self.__mzTargetFS = value

    @property
    def SN_thres(self):
        return self.__SN_thres

    @SN_thres.setter
    def SN_thres(self,value):
        self.__SN_thres=value

    @property
    def plotPath(self):
        return self.__resultsPath

    @property
    def resultsPath(self):
        return self.__resultsPath


    @resultsPath.setter
    def resultsPath(self, value):
        self.__resultsPath="./"
        if value!="":
            self.__resultsPath = value+"/"
            if not(os.path.exists(self.__resultsPath)):
                os.mkdir(self.__resultsPath)            

    @property
    def nameSufix(self):
        return self.__nameSufix

    @nameSufix.setter
    def nameSufix(self, value):
        self.__nameSufix = value
        if value!="":
            self.__nameSufix = "_"+value

    @property
    def namePrefix(self):
        return self.__namePrefix

    @namePrefix.setter
    def namePrefix(self, value):
        self.__namePrefix=value
        if value!="":
            self.__namePrefix = value+"_"

    @property
    def completeExpIsotopicPattSection(self):
        return self.__completeExpIsotopicPattSection

    @completeExpIsotopicPattSection.setter
    def completeExpIsotopicPattSection(self,value):
        self.__completeExpIsotopicPattSection=value

    @property
    def chromPeakMinNumOfPoints(self):
        return self.__chromPeakMinNumOfPoints

    @chromPeakMinNumOfPoints.setter
    def chromPeakMinNumOfPoints(self, value):
        self.__chromPeakMinNumOfPoints=value
       

    @property
    def chromPeakMinNumOfPointsFS(self):
        return self.__chromPeakMinNumOfPointsFS

    @chromPeakMinNumOfPointsFS.setter
    def chromPeakMinNumOfPointsFS(self, value):
        self.__chromPeakMinNumOfPointsFS=max(value,3)

    @property
    def chromPeakMinNumOfPointsMS2(self):
        return self.__chromPeakMinNumOfPointsMS2

    @chromPeakMinNumOfPointsMS2.setter
    def chromPeakMinNumOfPointsMS2(self, value):
        self.__chromPeakMinNumOfPointsMS2=max(value,2)
        

    @property
    def numOfIsoSignals(self):
        return self.__numOfIsoSignals

    @numOfIsoSignals.setter
    def numOfIsoSignals(self,value):
        self.__numOfIsoSignals=value


    @property
    def probableFragmentIons(self):
        return self.__probableFragmentIons

    @probableFragmentIons.setter
    def probableFragmentIons(self,value):
        self.__probableFragmentIons=value
    

    def setIonSpecies(self,ionSpecies):
        if self.__currentIonSpeciesList.empty: return

        self.__currentIonSpecies=self.__currentIonSpeciesList[self.__currentIonSpeciesList.formula==ionSpecies].copy().reset_index(drop=True)
        
        return
            
        
    def getEIsotopicPatternSection(self):
        df=pd.DataFrame()
        self.__completeExpIsotopicPattSection=pd.DataFrame()

        if ( (not(self.__ms.empty)) & (not(self.__isotopicPattUpperBound is None)) ): 

            lowerIdx=np.abs(self.__ms.mz-self.__mzTarget).idxmin()
            upperIdx=np.abs(self.__ms.mz-self.__isotopicPattUpperBound).idxmin()
            upperIdx+=1
            df=self.ms[lowerIdx:upperIdx].copy()
            df=df.reset_index(drop=True)
            self.__completeExpIsotopicPattSection=df.copy()
            self.__accurateMass=df.mz[0]
            self.__rMassError=self.relativeError(self.__accurateMass,self.mzTarget)
            
        return df


    def getGEIsotopicPatternSection(self,inMZSpect,mzTarget,dMZTarget=10.0):
        df=pd.DataFrame()
        accurateMass="-"
        rMassError="-"
        if not(inMZSpect.empty): 

            lowerIdx=np.abs(inMZSpect.mz-mzTarget).idxmin()
            upperIdx=np.abs(inMZSpect.mz-(mzTarget+dMZTarget)).idxmin()
            upperIdx+=1
            df=inMZSpect[lowerIdx:upperIdx].copy()
            df=df.reset_index(drop=True)
            accurateMass=df.mz[0]
            rMassError=self.relativeError(accurateMass,mzTarget)
            
        return (df,accurateMass,rMassError)
    
            
            
    def setAdqParameters(self,MSfilter,polarity,mzAcc,mzTarget=None,resultsPath="./",nameSufix=""):
        self.__MSfilter=MSfilter
        self.polarity=polarity
        self.__mzAcc=mzAcc
        self.__mzTarget=mzTarget
        self.__resultsPath=resultsPath

    def settedAdqParameters(self):
        return pd.DataFrame({"Parameter":["MSfilter","polarity","mzAcc","mzTarget","isotopicPattUpperBound","resultsPath"],
                      "Value":[self.__MSfilter,self.polarity,self.__mzAcc,self.__mzTarget,self.__isotopicPattUpperBound,self.__resultsPath]
                      }).set_index('Parameter')
               

    def getInstrumentalMethodInfo(self,rawFile=""):
        file_=rawFile
        if file_=="": file_=self.__MSfile

        if os.path.exists(file_):
            cmd_args=" '"+file_+"' -g instMethodInfo | grep -iv 'Closing'"
            os.system(self.__cmd+cmd_args)

    def getElutionInfo(self,isPlotShown=False):
        
        if self.__MSfile=="": return
        
        cmd_args=" '"+self.__MSfile+"' -g elutionInfo | grep -iv 'Closing'"
        proc=subprocess.Popen(self.__cmd+cmd_args,stdout=subprocess.PIPE,shell=True)
        self.__gradient=pd.DataFrame(columns=['time','A','B','C','D','flow','temperature'])

        for line in proc.stdout:
            __line=line.decode()
            if __line.find("ColumnOven.Temperature.Nominal")>0:
                temp=float(__line.split()[2])
            elif __line.find(" Flow ")>0:
                __solventComp=[]
                __line=__line.split()               
                if __line[0]=='Flow':
                    t=0.0
                    flow=float(__line[2])
                else:
                    t=float(__line[0])
                    flow=float(__line[3])
            elif __line.find(" %B ")>0:
                __solventComp=__solventComp+[float(__line.split()[2])]
            elif __line.find(" %C ")>0:
                __solventComp=__solventComp+[float(__line.split()[2])]
            elif __line.find(" %D ")>0:
                __solventComp=__solventComp+[float(__line.split()[2])]
                __solventComp=[t]+[100.0-sum(__solventComp)]+__solventComp+[flow,temp]
                self.__gradient.loc[len(self.__gradient)]=__solventComp

        self.__gradient=self.__gradient.drop_duplicates().reset_index(drop=True)

        if isPlotShown:
            self.plotElutionGradient()
                
        return self.__gradient.drop_duplicates().reset_index(drop=True)


    def plotElutionGradient(self):

        if self.__gradient.empty: return

        plt.rcdefaults()
        plt.clf()
        plt.plot('time',"A",data=self.__gradient,alpha=0.3,label="A")
        plt.plot('time',"B",data=self.__gradient,color='red',alpha=0.7,label="B")
        plt.xlabel('time')
        plt.ylabel('% Solvent')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.rcdefaults()
        plt.clf()
        
    

    def getLCGradientOrganicComposition(self,tR=-1.0,gradient=None):

        _tR=tR
        if tR<0.0:
            _tR=self.tR
            if isinstance(_tR,type(None)): return 0.0
                          

        
        __gradient=gradient
        if not(isinstance(gradient,pd.core.frame.DataFrame)):
            __gradient=self.gradient
        Bcomp=None
        if (__gradient.time.min()<=_tR) & (__gradient.time.max()>=_tR):

            locIdx=abs(__gradient.time-_tR).idxmin()
            __gradient=__gradient.iloc[[locIdx,locIdx+np.sign(_tR-__gradient.time.iloc[locIdx])]].sort_values('time')
            m=__gradient.iloc[1]-__gradient.iloc[0]
            m=m.B/m.time
            Bcomp=__gradient.B.iloc[1]+m*(_tR-__gradient.time.iloc[1])
        elif __gradient.time.max()<_tR:
            Bcomp=__gradient.B.iloc[-1]
        else:
            Bcomp=__gradient.B.iloc[0]
        return Bcomp
            
    

    def __getAdqMode(self):


        if self.__MSfile=="": return

        gdF=False
        cmd_args=" '"+self.__MSfile+"' -g instMethodInfo | grep -iv 'Closing'"
        proc=subprocess.Popen(self.__cmd+cmd_args,stdout=subprocess.PIPE,shell=True)
        self.__adqParametersFile=self.__MSfile
        
        experiment=""
        self.__adqParameters=pd.DataFrame()
        for line in proc.stdout:
            __line=line.decode().replace('\r\n','')
            if (__line.find("Setup")>0) & gdF: break                

            if __line.find("Experiment")>0:
                gdF=True

            elif gdF:

                if __line.find("General")==0:
                    experiment=prevLine

                elif len(__line.split("       ")) > 1:
                    adqData=__line.split("       ")
                    paramValue=adqData[-1].replace(",",".").replace("m/z","").replace("ce:","").strip()

                    if adqData[0].find("vDIA isolation range")==0:
                        paramValue=paramValue.split("to")
                        paramValue=f"{(float(paramValue[1])-float(paramValue[0]))/2};{(float(paramValue[1])+float(paramValue[0]))/2}"

                    
                    self.__adqParameters=pd.concat([self.__adqParameters,pd.DataFrame({"Parameter":[adqData[0]],"Value":[paramValue],"Experiment":[experiment]})],sort=False,ignore_index=True)

            prevLine=__line.strip()
                            

        return self.__adqParameters

    
        
    def getSampleInfo(self,rawFile=""):
        file_=rawFile
        if file_=="": file_=self.__MSfile

        if os.path.exists(file_):
            cmd_args=" '"+file_+"' -g sampleInfo"
            os.system(self.__cmd+cmd_args)

    def getEvents(self,rawFile="",setEvents=False):
        file_=rawFile
        if file_=="": file_=self.__MSfile       
        
        df=pd.DataFrame()
        if os.path.exists(file_):
            cmd_args=" '"+file_+"' -g eventsInfo | grep -iv 'Closing'"
            proc=subprocess.Popen(self.__cmd+cmd_args,stdout=subprocess.PIPE,shell=True)
            df = pd.read_csv(proc.stdout,sep=";",header=None)
            df.columns=["nScan","tR","mzFilterEvent","tic","lowmz","highmz","basePeakAbundance","mzBasePeak"]
            self.__eventsFile=file_

            if setEvents:
                 self.__events=df
                 self.__eventsState = True
            
        else:
            if self.__isVerbose:
                print("E.(RawReader): the file does not exit")
        return df

        
    def getAnalysisInfo(self,rawFile=""):
        file_=rawFile
        if file_=="": file_=self.__MSfile

        if os.path.exists(file_):
            cmd_args=" '"+file_+"' -g rawAnalysisInfo"
            os.system(self.__cmd+cmd_args)

    def getFilters(self,rawFile=""):

        file_=rawFile
        if file_=="": file_=self.__MSfile

        dfFilters=pd.DataFrame()
        if os.path.exists(file_):       
            cmd_args=" '"+file_+"' -g filtersInfo | grep -iv 'Closing' | sed 's/Filter [0-9]\\+: //g'"
            proc=subprocess.Popen(self.__cmd+cmd_args,stdout=subprocess.PIPE,shell=True)
            dfFilters = pd.read_csv(proc.stdout,sep=";",header=None)
            
            dfFilters=dfFilters.replace(" Full ms \["," Full ms1 -@- [",regex=True)[0].str.split(" ",expand=True).drop(columns=[0],axis=1)

        
            if len(dfFilters.columns)==7:
                dfFilters[0]=pd.to_numeric(dfFilters[6].str.split("@",expand=True)[0], errors='coerce',downcast='float')
                dfFilters[4]=dfFilters[[3,4,5,6]].agg(" ".join,axis=1)
                dfFilters[6]=dfFilters[6].str.split("@",expand=True)[1]
                dfFilters=dfFilters.replace("ms1 -@-"," ms",regex=True).replace("-@-","",regex=True)
                dfFilters.columns=['polarity','spectrumDataType','ionization','filter','MSn','CE','massRange','targetMass']
            elif len(dfFilters.columns)==8:
                dfFilters[0]=pd.to_numeric(dfFilters[7].str.split("@",expand=True)[0], errors='coerce',downcast='float')
                dfFilters[5]=dfFilters[[3,4,5,6,7]].agg(" ".join,axis=1)
                dfFilters[7]=dfFilters[7].str.split("@",expand=True)[1]
                dfFilters=dfFilters.replace("ms1 -@-"," ms",regex=True).replace("-@-","",regex=True)
                dfFilters.columns=['polarity','spectrumDataType','ionization','SID','filter','MSn','CE','massRange','targetMass']
                
            dfFilters=dfFilters.drop_duplicates().sort_values("MSn").reset_index(drop=True)
            
        return dfFilters


    def getScan(self,nScan,isCentroid=True,isPlotShown=False,isPlotSaved=False):

        if self.__MSfile=="":
            return

        if os.path.exists(self.__MSfile):

            processingType="centroid"
            if( not(isCentroid) ): processingType="profile"  
            cmd_args=" '"+self.__MSfile+"' -g scanData -n "+str(nScan)+" -t "+processingType+" | sed 's:,:;:g' | grep -iv 'Closing' | grep '^[0-9]'"
            proc=subprocess.Popen(self.__cmd+cmd_args,stdout=subprocess.PIPE,shell=True)

            try:
                df = pd.read_csv(proc.stdout,sep=";",header=None)          
            except:
                return pd.DataFrame()

            if not(df.empty):
                if df.shape[1]==7:
                    df.columns=['num','mz','intensity','resolution','charge','baseline','noise']
                elif df.shape[1]==3:
                    df.columns=['num','mz','intensity']
                else:
                    return pd.DataFrame()
                if isPlotSaved | isPlotShown:
                    plt.rcdefaults()
                    plt.clf()
                    plt.stem(df.mz,df.intensity,markerfmt='none',basefmt="gray")
                    plt.xlabel('m/z')
                    plt.ylabel('Intensity')
                    pltName="ms_"+self.fileName
                    plt.tight_layout()
                    if isPlotSaved:
                        plt.savefig(os.path.join(self.resultsPath,pltName+".png"))
                    if isPlotShown:
                        plt.show()
                    plt.clf()
                return df
            else:
                return pd.DataFrame()

    def getXICRidgeInfo(self):
        if (os.path.exists(self.__MSfile)) & ( not(self.__mzTarget is None) ):
            __chromSignal=self.getChromatogram()
            if __chromSignal.empty: return None
            __ridgeInfo=__chromSignal.iloc[[__chromSignal['intensity'].idxmax()]].reset_index(drop=True)
            return __ridgeInfo
        return None

    
    def getExperimentalIsotopicPatternSection(self,isPlotShown=False):

        if (os.path.exists(self.__MSfile)) & ( not(self.__mzTarget is None) ):
            __ridgeInfo=self.getXICRidgeInfo()
            if isinstance(__ridgeInfo,type(None)): return None
            if self.__events.empty: self.__events=self.getEvents().copy()
            __ridgeEventInfo=self.__events[self.__events.tR==__ridgeInfo.time.iloc[0]].iloc[0]
            __massSpectrum=self.getScan(__ridgeEventInfo.nScan)
            __isoPatternSection=__massSpectrum[(__massSpectrum.mz>=(self.mzTarget*(1.0-self.mzAcc/1E6))) & (__massSpectrum.mz<=(self.mzTarget+7.0))].reset_index(drop=True)[['mz','intensity']]
            __isoPatternSection['rintensity']=__isoPatternSection.intensity/__isoPatternSection.intensity.max()

            if isPlotShown:
                plt.rcdefaults()
                plt.clf()
                emarkerline, estemlines, ebaseline=plt.stem(__isoPatternSection.mz,__isoPatternSection.intensity,basefmt="none",linefmt="r-")
                estemlines.set_alpha(0.25)
                ebaseline.set_color('gray')
                ebaseline.set_linewidth(0.1)
                emarkerline.set_markersize(3)
                plt.xlabel('m/z')
                plt.ylabel('Relative Intensity')
                plt.tight_layout()
                plt.show()
                plt.rcdefaults()
                plt.clf()
                
            return __isoPatternSection
        return pd.DataFrame()
                      

    def getChromatogram(self,isPlotShown=False,isPlotSaved=False,mzTarget=None,title="",minNPoints=3):
        df = pd.DataFrame()
        if self.__MSfile=="":
            return df

        __mzTarget=mzTarget
        if __mzTarget is None:
            __mzTarget=self.__mzTarget

        if __mzTarget is None:
            return df
        
        if os.path.exists(self.__MSfile):
            cmd_args='"'+self.__MSfile+'" -g chromatogram  '+'-trace massRange -mzFilter "'+self.__polarity.strip()+" "+self.__MSfilter+'" -mzAcc '+str(self.__mzAcc)+' -massRange '+' "['+str(__mzTarget*(1.0-self.__mzAcc/1E6))+';'+str(__mzTarget*(1.0+self.__mzAcc/1E6))+']" | grep -iv "Closing"'

            proc=subprocess.Popen(self.__cmd+cmd_args,stdout=subprocess.PIPE,shell=True)

            try:
                df = pd.read_csv(proc.stdout,sep=";",header=None)          
                df.columns=['time','intensity']
            except:
                return pd.DataFrame()

            if (df[df['intensity'] >= 1E-6]).shape[0]<minNPoints:
                return pd.DataFrame()


            if isPlotSaved | isPlotShown:
                plt.rcdefaults()
                plt.clf()
                plt.plot(df.time,df.intensity)
                plt.xlabel('time / min')
                plt.ylabel('Intensity')
                plt.title(title)
                pltName="xic_"+self.fileName
                plt.tight_layout()
                if isPlotSaved:
                    plt.savefig(os.path.join(self.resultsPath,pltName+".png"))
                if isPlotShown:
                    plt.show()
                plt.clf()
        return df

    def getMSnChromatogramByMZ(self,mzTarget,filterNum=0,isPlotShown=True,showFilters=False,isPlotSaved=False,title="",minNPoints=2,xlims=[],msfiltersDF=None):
        df = pd.DataFrame()
        if self.__MSfile=="":
            return df

        __mzTarget=mzTarget
        if __mzTarget is None:
            return df
        
        if os.path.exists(self.__MSfile):
            msnFilters=msfiltersDF
            if isinstance(msfiltersDF,type(None)): msnFilters=self.getFilters()
            if showFilters:
                display(msnFilters)
                
            _nF=filterNum%len(msnFilters)
            
            cmd_args='"'+self.__MSfile+'" -g chromatogram  '+'-trace massRange -mzFilter "'+msnFilters.polarity[_nF]+" "+msnFilters["filter"][_nF]+'" -mzAcc '+str(self.__mzAcc)+' -massRange '+' "['+str(__mzTarget*(1.0-self.__mzAcc/1E6))+';'+str(__mzTarget*(1.0+self.__mzAcc/1E6))+']" | grep -iv "Closing"'

            proc=subprocess.Popen(self.__cmd+cmd_args,stdout=subprocess.PIPE,shell=True)

            try:
                df = pd.read_csv(proc.stdout,sep=";",header=None)          
                df.columns=['time','intensity']
            except:
                return pd.DataFrame()

            if (df[df['intensity'] >= 1E-6]).shape[0]<minNPoints:
                return pd.DataFrame()


            if isPlotSaved | isPlotShown:
                plt.rcdefaults()
                plt.clf()
                _df=df.copy()
                if len(xlims)>1:
                    _df=_df[_df.time>=xlims[0]]
                    _df=_df[_df.time<=xlims[1]]

                plt.plot(_df.time,_df.intensity)
                plt.xlabel('time / min')
                plt.ylabel('Intensity')
                plt.title(title)
                pltName="xic_"+self.fileName
                
                plt.tight_layout()
                if isPlotSaved:
                    plt.savefig(os.path.join(self.resultsPath,pltName+".png"))
                if isPlotShown:
                    plt.show()
                plt.clf()
        return df
    

    def getFragmentsIonsOverlay(self,mzMS1,ms2Fragments,tR=None,ms2FilterNum=None,areXICPloted=True,isScaled=False,dtR=0.1,ms2NomFactor=1.0):

        __filters=self.getFilters()
        if __filters.empty: return pd.DataFrame()
                
        xicVault=pd.DataFrame()
        _xic=self.getMSnChromatogramByMZ(mzTarget=mzMS1,filterNum=0,isPlotShown=False,msfiltersDF=__filters)
        _tR=tR

        if _xic.empty: return pd.DataFrame()


        if isinstance(tR,type(None)):
            _tR=_xic.time[_xic.intensity.idxmax()]

        tRLims=[_tR-dtR,_tR+dtR]
        _xic=_xic[(_xic.time>=tRLims[0]) & (_xic.time<=tRLims[1])]
        if ( _xic.empty | (not(any(_xic.intensity>=1E-6))) ): return pd.DataFrame()


        _xic["msLevel"]="MS1"
        _xic["fragment"]=0
        _xic["mz"]=mzMS1
        _xic["tRmax (FS)"]=_tR
        _xic["filter"]=__filters["filter"][0]
        _xic["normFactor"]=1
        xicVault=pd.concat([xicVault,_xic],sort=False,ignore_index=True)
        foundFragments=[]


        if isinstance(ms2FilterNum,type(None)):
            filtersToSearch=list(range(1,len(__filters)))
        else:
            nFilters=len(__filters)
            _ms2FilterNum=ms2FilterNum%nFilters
            if _ms2FilterNum==0:_ms2FilterNum=1
            filtersToSearch=[_ms2FilterNum]



            
        i=0
        for fidx in filtersToSearch:    
            for fragmentMZ in ms2Fragments:
                _xic=self.getMSnChromatogramByMZ(mzTarget=fragmentMZ,filterNum=fidx,isPlotShown=False,msfiltersDF=__filters)
                if not(_xic.empty):
                    _xic=_xic[(_xic.time>=tRLims[0]) & (_xic.time<=tRLims[1])]
                    if not(_xic.empty):
                        if any(_xic.intensity>=1E-6):
                            i+=1
                            foundFragments=foundFragments+[i]
                            _xic["msLevel"]="MS2"
                            _xic["fragment"]=i
                            _xic["mz"]=fragmentMZ
                            _xic["tRmax (FS)"]=_tR
                            _xic["filter"]=__filters["filter"][fidx]
                            _xic["normFactor"]=ms2NomFactor
                            xicVault=pd.concat([xicVault,_xic],sort=False,ignore_index=True)

        if xicVault.empty: return pd.DataFrame()
        yLab=""
        rdf=xicVault.copy()

        if areXICPloted:
            plt.rcdefaults()
            plt.clf()

            if isScaled:
                yLab="Normalized "
                xicVault.intensity=(xicVault.intensity/xicVault.groupby("fragment")["intensity"].transform('max'))/xicVault.normFactor

            fig, ax = plt.subplots()
            pltData=xicVault[xicVault.fragment==0]
            ax.plot(pltData.time,pltData.intensity, label=f"MS1: {mzMS1}", linewidth=3)
            for i in foundFragments:
                pltData=xicVault[xicVault.fragment==i].copy().reset_index(drop=True)
                cfilt=pltData["filter"][0]
                ax.plot(pltData.time,pltData.intensity, label=f"MS2: {pltData.mz[0]} ({cfilt})", linewidth=0.5)

            ax.set_ylabel(f"{yLab}Intensity")
            ax.set_xlabel('time')

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True,fontsize=8)
            plt.tight_layout()
            plt.show()

        return rdf
    

    def getXICArray(self,width=75,pkHog=True,visualHog=False,tRLims=None,maxtRThres=None,filterMask=False):

        if self.__MSfile=="":
            return

        if self.__mzTarget is None:
            return

        
        if os.path.exists(self.__MSfile):


            cmd_args='"'+self.__MSfile+'" -g chromatogram  '+'-trace massRange -mzFilter "'+self.__polarity.strip()+" "+self.__MSfilter+'" -mzAcc '+str(self.__mzAcc)+' -massRange '+' "['+str(self.__mzTarget)+';0.0]" | grep -iv "Closing"'

            proc=subprocess.Popen(self.__cmd+cmd_args,stdout=subprocess.PIPE,shell=True)

            try:

                df = pd.read_csv(proc.stdout,sep=";",header=None)          
                df.columns=['time','intensity']

                if df.intensity.max()<self.__intensityThres:
                    return pd.DataFrame()

                if isinstance(tRLims,list):
                    df=df[(df.time>=tRLims[0]) & (df.time<=tRLims[1])].reset_index(drop=True)
                    if filterMask:
                        df.intensity[df.time<self.filterMaskLims[0]]=0
                        df.intensity[df.time>self.filterMaskLims[1]]=0

                if isinstance(maxtRThres,list):
                    tRmax=df.time.iloc[df.intensity.idxmax()]
                    if ( (tRmax<=maxtRThres[0]) | (tRmax>=maxtRThres[1]) ):
                        return pd.DataFrame()

            except:
                return pd.DataFrame()

            if (df[df['intensity'] >= 1E-6]).shape[0]<self.__chromPeakMinNumOfPoints:
                return pd.DataFrame()

            plt.rcdefaults()
            plt.clf()
            plt.rcParams['ytick.labelleft']=False
            plt.rcParams['xtick.labelbottom']=False
            plt.rcParams['xtick.bottom']=False
            plt.rcParams['ytick.left']=False
            plt.rcParams['axes.linewidth']=False
            plt.plot(df.time,df.intensity)
            plt.tight_layout()
            membuf=io.BytesIO()
            plt.savefig(membuf,format="png")
            membuf.seek(0)
            plt.clf()
            plt.rcdefaults()
            pkImage=np.array(Image.open(membuf))
            membuf.close()
            pkImage=np.dot(pkImage[...,:3],[0.299,0.587,0.114]) 
            pkImage=resize(pkImage,(width,width))/255

            if pkHog:

                orientations_=6
                cells_per_block_=4
                pixels_per_cell_=12

                if not(visualHog):

                    pkImage= hog(
                        pkImage, pixels_per_cell=(pixels_per_cell_,pixels_per_cell_), 
                        cells_per_block=(cells_per_block_, cells_per_block_), 
                        orientations=orientations_,  
                        block_norm='L2-Hys',
                        feature_vector=True)
                else:
                    hogF,pkImage= hog(
                        pkImage, pixels_per_cell=(pixels_per_cell_,pixels_per_cell_), 
                        cells_per_block=(cells_per_block_, cells_per_block_), 
                        orientations=orientations_,
                        visualize=True,
                        block_norm='L2-Hys')
                    return (hogF,df,pkImage)

        return (pkImage,df)
    
       
    def plotXIC(self,isPlotShown=True,areAxisShown=True,areDataSaved=False,isXICSaved=False,title="",plot2dtR=None,U_tR=0.0):

        if not(self.xic.empty):
            fileName=self.fileName+"_xic"
            if areDataSaved:
                self.xic.to_csv(os.path.join(self.resultsPath,fileName+".tsv"),sep="\t",index=None)
            plt.rcdefaults()
            plt.clf()
            plt.rcParams['ytick.labelleft']=areAxisShown
            plt.rcParams['xtick.labelbottom']=areAxisShown
            plt.rcParams['xtick.bottom']=areAxisShown
            plt.rcParams['ytick.left']=areAxisShown
            plt.rcParams['axes.linewidth']=areAxisShown
            plt.plot('time','intensity',data=self.xic,linewidth=0.85,label='_Hidden')

            if plot2dtR!=None:
                __dispXic=self.xic.copy()
                __idxs=__dispXic[(__dispXic.time<self.tR-0.1) | (__dispXic.time>self.tR+0.1)].index.to_list()
                __dispXic.loc[__idxs,"intensity"]=0.0
                __dispXic['time']=__dispXic['time']+plot2dtR
                __dispXic=__dispXic[(__dispXic['time']<=self.xic.time.max()) & (__dispXic['time']>=self.xic.time.min()) ]
                plt.plot('time','intensity','--',data=__dispXic,color='gray',linewidth=0.85, alpha = 0.85,label="RT reference")


                if U_tR>0.0:
                    tRi__=self.tR+plot2dtR
                    y=norm.pdf(__dispXic['time'],tRi__, U_tR/1.96)
                    y=(y/y.max())*__dispXic.intensity.max()
                    plt.fill_between(__dispXic['time'], y,alpha=0.05,color='green')
                    plt.fill_between(__dispXic['time'], y,alpha=0.15,color='green',where=(__dispXic['time']>(tRi__-U_tR))&(__dispXic['time']<(tRi__+U_tR)) )
                    
                    plt.errorbar([self.tR+plot2dtR],[__dispXic.intensity.max()/2],xerr=U_tR, capsize=4,
                                 marker='o', markersize=6,linestyle='--',
                                 color='orange',linewidth=0.85, alpha = 0.25)

                  

                    
                del __dispXic
                del __idxs
                plt.legend()

            title_=f"{title}\nm/z: {round(self.mzTarget,5)} Th"
            
            if areAxisShown:
                plt.xlabel('time / min')
                plt.ylabel('Intensity')
                plt.title(title_)

            plt.tight_layout()

                
            if isXICSaved:
                plt.savefig(os.path.join(self.resultsPath,fileName+".png"))
            if isPlotShown:
                plt.show()
           
            plt.clf()
            plt.rcdefaults()
        return

        
    def plotMassSpectrum(self,isPlotShown=True,isSectionShown=False,areAxisShown=True,areDataSaved=False,isMSSaved=False,title="",appendMSdata=None):

        if not(self.ms.empty):
            fileName=self.fileName+"_ms"
            df=self.ms
            if isSectionShown: df=self.__completeExpIsotopicPattSection

            if areDataSaved:
                df.to_csv(os.path.join(self.resultsPath,fileName+".tsv"),sep="\t",index=None)
            plt.rcdefaults()
            plt.clf()
            plt.rcParams['ytick.labelleft']=areAxisShown
            plt.rcParams['xtick.labelbottom']=areAxisShown
            plt.rcParams['xtick.bottom']=areAxisShown
            plt.rcParams['ytick.left']=areAxisShown
            plt.rcParams['axes.linewidth']=areAxisShown
            markerline, stemlines, baseline=plt.stem('mz','intensity',data=df,markerfmt='none',basefmt="none",linefmt='blue')
            stemlines.set_alpha(0.25)

            if not(appendMSdata is None):
                plt.stem('mz','intensity',data=appendMSdata,markerfmt='none',basefmt="none",linefmt='r')
            
            if not(isSectionShown):
                plt.axvline(x=self.__completeExpIsotopicPattSection['mz'].iloc[-1],
                            ymin=0.0,ymax=1.0,color='gray',
                            linestyle = ':', alpha = 0.5)
                plt.axvline(x=self.__completeExpIsotopicPattSection['mz'].iloc[0],
                            ymin=0.0,ymax=1.0,color='gray',
                            linestyle = ':', alpha = 0.5)


            if areAxisShown:
                title_=f"m/z: {round(self.mzTarget,5)} Th"
                if title!="": title_=title

                
                plt.xlabel('m/z (Th)')
                plt.ylabel('Intensity')
                plt.title(title_)

            plt.tight_layout()
                
            if isMSSaved:
                plt.savefig(os.path.join(self.resultsPath,fileName+".png"))
            if isPlotShown:
                plt.show()
           
            plt.clf()
            plt.rcdefaults()
        return

    def relativeError(self, measurement, reference,scale=1E6):      
        return (measurement-reference)/reference*scale # ppm


    def __setNearScanParameters(self,ms1ScanNum,candidateFilters,tR_thres=None):
        __tR_thres=tR_thres

        if isinstance(__tR_thres,type(None)):
            __tR_thres=list(abs(np.array(self.filterMaskLims)-self.__events.tR[ms1ScanNum-1]))

        self.__getNearestFilter(candidateFilters,self.mzTargetFS,
                                ms1ScanNum,
                                tR_thres=__tR_thres)
        return
    
    def __getNearestFilter(self,candidateFilters,mzPrecursor,ms1Scan,tR_thres=0.1):
        self.adqExperiments
        self.__nearestMS2ScanIdx=0
        self.__nearMS2Scans=[]
        self.__nearMS2Events=pd.DataFrame()
        self.__nearMS2Filters=pd.DataFrame()
        adqParams=self.__adqParameters.copy()
        __tR_thres=tR_thres

        if isinstance(ms1Scan,type(None)): return pd.DataFrame()
        if not(isinstance(__tR_thres,list)):__tR_thres=[tR_thres]*2

        if not(adqParams[adqParams.Parameter=="Isolation window"].empty):
            isolationWin=float(adqParams[adqParams.Parameter=="Isolation window"].iloc[0].Value)

            self.__nearMS2Filters=candidateFilters[((candidateFilters.targetMass-isolationWin)<=mzPrecursor) & ((candidateFilters.targetMass+isolationWin)>=mzPrecursor)].reset_index(drop=True).copy()


        elif not(adqParams[adqParams.Parameter.str.find("vDIA isolation range")==0].empty):

            isolationWin=adqParams[adqParams.Parameter.str.find("vDIA isolation range")==0].Value.str.split(";",expand=True)
            isolationWin[0]=pd.to_numeric(isolationWin[0])
            isolationWin[1]=pd.to_numeric(isolationWin[1])
            isolationWin=isolationWin[(isolationWin[1]-isolationWin[0]<=mzPrecursor) & (isolationWin[1]+isolationWin[0]>=mzPrecursor)]
            filtersRE="|".join([str(mz) for mz in isolationWin[1].to_list()])
            self.__nearMS2Filters=candidateFilters[candidateFilters['filter'].str.findall(r""+filtersRE).apply(len)>0].copy()

        else:
            return pd.DataFrame()

       
        if self.__nearMS2Filters.empty: return pd.DataFrame()
        filtersRE="|".join(self.__nearMS2Filters['filter'].to_list())
        tR_ref=self.__events.tR[ms1Scan-1]-__tR_thres[0] 
        startIdx=abs(self.__events.tR-tR_ref).idxmin()
        tR_ref=self.__events.tR[ms1Scan-1]+__tR_thres[1] 
        endIdx=abs(self.__events.tR-tR_ref).idxmin()+1
        filters_=self.__events.loc[startIdx:endIdx].copy()
        filters_=filters_[filters_.mzFilterEvent.str.findall(rf"FTMS [{self.__polarity}] ").apply(len)>0]
        self.__nearMS2Scans=filters_[filters_.mzFilterEvent.str.findall(r""+filtersRE).apply(len)>0].nScan.to_list()

        if len(self.__nearMS2Scans)>0:
            filters_=filters_[filters_.mzFilterEvent.str.findall(r""+filtersRE).apply(len)>0].copy()
            self.__nearMS2Events=filters_.copy()
            self.__nearestMS2ScanIdx=np.argmin(abs(filters_.tR-self.__events.tR[ms1Scan-1]))

            filters_=filters_.mzFilterEvent.iloc[self.__nearestMS2ScanIdx].split(" ")[-2]
            filters=self.__nearMS2Filters[self.__nearMS2Filters['filter'].str.findall(r""+filters_).apply(len)>0].copy().reset_index(drop=True)
            self.__ms2Filter=filters.copy()
        else:
            self.__ms2Filter="--"
            return pd.DataFrame()
            
        return filters


    def __setAdqFilters(self,mzPrecursor,ms1Scan=None,adqType="ms1",mzAcc=5.0,polarity=None,tR_thres=0.1):

        self.__isAdqFilterSetted=False
        if len(re.findall("[+-]",str(polarity)))==0: return
        if self.__MSfile=="": return 

        self.filters
        self.events=self.__MSfile

        self.__candidateFilters=self.__filters.copy()
        if polarity!=None:
            self.__candidateFilters=self.__candidateFilters[self.__candidateFilters.polarity==polarity].reset_index(drop=True)
            if self.__candidateFilters.empty: return


        if adqType=="ms1":
            self.__candidateFilters=self.__candidateFilters[self.__candidateFilters.MSn=="ms1"].reset_index(drop=True)

            if self.__candidateFilters.empty: return

            self.setAdqParameters(self.__candidateFilters.iloc[0]['filter'],self.__candidateFilters.iloc[0]['polarity'],mzAcc,mzPrecursor)
            self.__spectrumDataType=self.__candidateFilters.iloc[0]['spectrumDataType']
            self.chromPeakMinNumOfPoints=self.chromPeakMinNumOfPointsFS
        elif adqType=="ms2":
            self.mzTargetPrecursor=mzPrecursor
            self.__candidateFilters=self.__getNearestFilter(self.__candidateFilters,mzPrecursor,ms1Scan,tR_thres=tR_thres)
            if self.__candidateFilters.empty: return
            
            self.chromPeakMinNumOfPoints=self.chromPeakMinNumOfPointsMS2
            self.setAdqParameters(self.__candidateFilters.iloc[0]['filter'],self.__candidateFilters.iloc[0]['polarity'],mzAcc)
            self.__spectrumDataType=self.__candidateFilters.iloc[0]['spectrumDataType']
        else:
            return
        
        self.__isAdqFilterSetted=True
    
    
    def searchIon(self,mzProduct=None,tR_lims=None,maxtR_thres=None,ms1Scan=None,filterMask=None):


        if self.__isAdqFilterSetted:

            if isinstance(mzProduct,float): self.mzTarget=mzProduct
            
            chromDF=self.getXICArray(tRLims=tR_lims,maxtRThres=maxtR_thres,filterMask=filterMask)

            if isinstance(chromDF,tuple):
                tRmax=chromDF[1].time.iloc[chromDF[1].intensity.idxmax()]
                scanNum2tRmax="-"
                mzSpec=pd.DataFrame()

                if not(self.__events.empty):
                    scanNum2tRmax=self.eventsDF[self.eventsDF.tR==tRmax].nScan.iloc[0]
                    mzSpec=self.getScan(scanNum2tRmax,self.__spectrumDataType)

                return (chromDF[0],chromDF[1],mzSpec,tRmax,scanNum2tRmax,self.__candidateFilters.iloc[0]['filter'])
            else:
                return

    def setXICFilterMask(self,xicTemplate,invHfraction=5,p=0.99,tR_lims=None):

        if isinstance(tR_lims,list):
            self.__filterMaskLims=tR_lims
            return

        if isinstance(xicTemplate,pd.core.frame.DataFrame):
            if xicTemplate.empty: return
            
            filterMask=xicTemplate[['time','intensity']].copy()
            filterMask.intensity=filterMask.intensity/filterMask.intensity.max()
            tRmax=filterMask.time[filterMask.intensity.idxmax()]

            if tRmax>=filterMask.time.max():
                self.__filterMaskLims=[tRmax]*2
                return
            
            filterMaskTmp=filterMask[filterMask.time>=tRmax]
            idxHfrac=(abs(filterMaskTmp.intensity-1/invHfraction)*abs(tRmax-filterMaskTmp.time)).sort_values()
            r_lim=tRmax
            if len(idxHfrac)>1:
                idxHfrac=idxHfrac.index[1]
                x_Hfrac_R=filterMaskTmp.time[idxHfrac]-tRmax
                zR=np.log(invHfraction)/x_Hfrac_R
                r_lim=max(tRmax+0.15,tRmax-np.log(1.0-p)/zR)

            
            filterMaskTmp=filterMask[filterMask.time<=tRmax]
            idxHfrac=(abs(filterMaskTmp.intensity-1/invHfraction)*abs(tRmax-filterMaskTmp.time)).sort_values()
            l_lim=tRmax
            if len(idxHfrac)>1:
                idxHfrac=idxHfrac.index[1]           
                x_Hfrac_L=abs(filterMaskTmp.time[idxHfrac]-tRmax)            
                zL=np.log(invHfraction)/x_Hfrac_L
                l_lim=max(0.0,min(tRmax-0.15,tRmax+np.log(1.0-p)/zL))

            self.__filterMaskLims=[l_lim,r_lim]
            
        return


    def assessMS1_MS2_signals(self,ms1XIC,ms2XIC,thres=1.75,dInt=0.2):

        ms2XIC_masked=ms2XIC[(ms2XIC.time>=self.filterMaskLims[0]) &
                             (ms2XIC.time<=self.filterMaskLims[1])].copy()
                

        if (ms2XIC_masked[ms2XIC_masked['intensity'] >= 1E-6]).shape[0]>=self.__chromPeakMinNumOfPointsMS2:
            ms1XICSection=ms1XIC.copy()
            ms1XICSection=ms1XICSection[(ms1XICSection.time>=self.filterMaskLims[0]) &  (ms1XICSection.time<=self.filterMaskLims[1])]
            ms1SampledXICSection=ms1XICSection.copy()
            ms2XIC_masked_maxInt=ms2XIC_masked.intensity.max()
            ms2XIC_masked.intensity=ms2XIC_masked.intensity/ms2XIC_masked_maxInt
            nearIdx=[abs(ms1SampledXICSection.time-t).idxmin() for t in ms2XIC_masked.time] 
            ms1SampledXICSection=ms1SampledXICSection.loc[nearIdx]
            ms1SampledXICSection.intensity=ms1SampledXICSection.intensity/ms1SampledXICSection.intensity.max()    


            SS_max=len(ms2XIC_masked.intensity)*(dInt**2)
            SS=sum((ms2XIC_masked.intensity.to_numpy()-ms1SampledXICSection.intensity.to_numpy())**2)
            peakSimilarity=SS/SS_max
            p_=1.0-f.cdf(peakSimilarity,__ms2XICShape[0] , 1000, loc=0, scale=1)
            
            return (ms1SampledXICSection,p_,ms2XIC_masked,ms2XIC_masked_maxInt,ms1XICSection)
        
        return (pd.DataFrame(),0.0,pd.DataFrame(),0.0,pd.DataFrame())

    def assessMS2ProfileQuality(self,fragmentIons):
        __pMS1Q=fragmentIons.p_MS1_MS2.to_numpy()
        __nFrags=len(__pMS1Q)

        #x=3,p=0.95,b=1.0,-2/alpha=-2.0/(2.0*(x-b)/np.log(p/(1.0-p)))=-0.7361097447916098
        p__nFrags=1.0/(1.0+4.3588221998035515*np.exp(-1.472201883207457*__nFrags))
        return sum(__pMS1Q)/__nFrags*p__nFrags 



    def searchNontargetedProbableFragmentIons(self,scanIdx=None,p_thres=0.25,ms1ScanNum=None,tR_thres=None):
        self.__probableFragmentIonsXICs=pd.DataFrame()
        self.__probableFragmentIons=pd.DataFrame()
        
        if self.__currentFSXICPeak.empty:
            return
      
        __MSfilter=self.MSfilter
        if isinstance(ms1ScanNum,int):

            if ms1ScanNum in self.multPeaksInfo.nScan.to_list():
                __maxInt=self.__multPeaksInfo[self.__multPeaksInfo.nScan==ms1ScanNum].intensity.iloc[0]
                __tRmaxInt=self.__multPeaksInfo[self.__multPeaksInfo.nScan==ms1ScanNum].time.iloc[0]
                __xicMS1Template=self.__currentFSXICPeak[self.__currentFSXICPeak.intensity<=__maxInt].copy().reset_index(drop=True)
                tR_lims=tR_thres
                if isinstance(tR_thres,float): tR_lims=[__tRmaxInt-tR_thres,__tRmaxInt+tR_thres]
                self.setXICFilterMask(__xicMS1Template,tR_lims=tR_lims)
                __candidateFilters=self.filters.copy()      
                __candidateFilters=__candidateFilters[__candidateFilters.polarity==self.polarity].reset_index(drop=True)

                if not(__candidateFilters.empty):           
                    self.__setNearScanParameters(ms1ScanNum,__candidateFilters,tR_thres=tR_thres)
                else:
                    return
            else:
                return
        else:

            __xicMS1Template=self.__currentFSXICPeak.copy()
            if isinstance(tR_thres,float):
                __candidateFilters=self.filters.copy()
                __candidateFilters=__candidateFilters[__candidateFilters.polarity==self.polarity].reset_index(drop=True)
                if not(__candidateFilters.empty):
                    self.setXICFilterMask(__xicMS1Template,tR_lims=[self.tR-tR_thres,self.tR+tR_thres])
                    self.__setNearScanParameters(self.currMS1ScanNum2tRmax,__candidateFilters,tR_thres=tR_thres)
                else:
                    return
                
            else:

                self.setXICFilterMask(__xicMS1Template)
                __candidateFilters=self.filters.copy()
                __candidateFilters=__candidateFilters[__candidateFilters.polarity==self.polarity].reset_index(drop=True)
                if not(__candidateFilters.empty):
                    self.__setNearScanParameters(self.currMS1ScanNum2tRmax,__candidateFilters,tR_thres=tR_thres)
                else:
                    return                 

        if len(self.__nearMS2Scans)==0: return

        if isinstance(scanIdx,int):
            scanIdx_=scanIdx % len(self.__nearMS2Scans)
            _nearestMS2Scan=self.__nearMS2Scans[scanIdx_]
            ionsSearchList=self.getScan(_nearestMS2Scan)
            filters_=self.__nearMS2Events.mzFilterEvent.iloc[scanIdx_].split(" ")[-2]
            self.MSfilter=self.__nearMS2Filters[self.__nearMS2Filters['filter'].str.findall(r""+filters_).apply(len)>0].copy().reset_index(drop=True).iloc[0]['filter']
            self.__currentCE=float(self.__nearMS2Filters[self.__nearMS2Filters['filter'].str.findall(r""+filters_).apply(len)>0].copy().reset_index(drop=True).iloc[0].CE.replace("hcd",""))
        else:
            _nearestMS2Scan=self.__nearMS2Scans[self.__nearestMS2ScanIdx]
            ionsSearchList=self.getScan(_nearestMS2Scan)
            self.MSfilter=self.lastMS2Filter.iloc[0]['filter']
            self.__currentCE=float(self.lastMS2Filter.iloc[0].CE.replace("hcd",""))

        ionsSearchList=ionsSearchList[ionsSearchList.mz<=self.__mzTargetFS+5].reset_index(drop=True)
        ionsSearchList=ionsSearchList[ionsSearchList.intensity<__xicMS1Template.intensity.max()].reset_index(drop=True)
        ionsSearchList=ionsSearchList[ionsSearchList.intensity>=self.__intensityThres].reset_index(drop=True)
        ionsSearchList=ionsSearchList[ionsSearchList.intensity/ionsSearchList.noise>self.__SN_thres].sort_values(['intensity'],ascending=False).reset_index(drop=True)

        dtR=((self.filterMaskLims[1]-self.filterMaskLims[0])/2.0)*(1.0-1.0/np.sqrt(6))
        maxRegVal=[self.filterMaskLims[0]+dtR,self.filterMaskLims[1]-dtR]

        processed_fragments=Parallel(n_jobs=self.__njobs)(delayed(assessFragments)(mz,
                                                                                   self.__MSfile,
                                                                                   self.__polarity,
                                                                                   self.__MSfilter,
                                                                                   self.__mzAcc,
                                                                                   self.__cmd,
                                                                                   self.filterMaskLims,
                                                                                   p_thres,
                                                                                   __xicMS1Template,
                                                                                   ionsSearchList.intensity.iloc[j],
                                                                                   self.chromPeakMinNumOfPointsMS2,
                                                                                   isChi2Metric=self.fragmentationChi2Metric,
                                                                                   dInt=self.fragmentationXIC_dInt,maxReg=maxRegVal) for j,mz in enumerate(ionsSearchList['mz']))

        k=0
        for fragment in processed_fragments:
            if not(fragment[0].empty):
                k+=1
                fragment[0]['mzPrecursor']=self.mzTargetFS
                self.__probableFragmentIons=pd.concat([self.__probableFragmentIons,fragment[0]],sort=False,ignore_index=True)
                fragment[1]['CE']=self.__currentCE
                fragment[1]['fragmentNum']=k
                self.__probableFragmentIonsXICs=pd.concat([self.__probableFragmentIonsXICs,fragment[1]],sort=False,ignore_index=True)

        if not(self.__probableFragmentIonsXICs.empty):
            ms1XICSection=__xicMS1Template.copy()
            ms1XICSection=ms1XICSection[(ms1XICSection.time>=self.filterMaskLims[0]) &  (ms1XICSection.time<=self.filterMaskLims[1])]
            ms1XICSection['mz']=self.mzTargetFS
            ms1XICSection['label']="ms1Template"
            ms1XICSection['p']=""
            ms1XICSection['intMax']=""
            ms1XICSection['CE']=self.__currentCE
            ms1XICSection['fragmentNum']=0
            self.__probableFragmentIonsXICs=pd.concat([ms1XICSection,self.__probableFragmentIonsXICs],sort=False,ignore_index=True)

        if not(self.__probableFragmentIons.empty):
            self.__probableFragmentIons=self.__probableFragmentIons.sort_values(['mz']).reset_index(drop=True)
        self.MSfilter=__MSfilter 
        return 


    def searchTargetProbableFragmentIons(self,targetMZ,polarity,ms2_tR_Lims,ms2_maxtRErr,mzAcc,ionsSearchList=None,ms1Scan=None,filterMask=False,tR_thres=0.1,p_thres=0.25):
        self.__probableFragmentIonsXICs=pd.DataFrame()
        self.__probableFragmentIons=pd.DataFrame()
        if self.__currentFSXICPeak.empty:
            return
        
        __MSfilter=self.MSfilter 
        __xicMS1Template=self.__currentFSXICPeak.copy()
        if isinstance(tR_thres,float):
            __candidateFilters=self.filters.copy()
            __candidateFilters=__candidateFilters[__candidateFilters.polarity==self.polarity].reset_index(drop=True)
            if not(__candidateFilters.empty):
                self.setXICFilterMask(__xicMS1Template,tR_lims=[self.tR-tR_thres,self.tR+tR_thres])
                self.__setNearScanParameters(self.currMS1ScanNum2tRmax,__candidateFilters,tR_thres=tR_thres)
            else:
                return

        else:
            self.setXICFilterMask(__xicMS1Template)
            __candidateFilters=self.filters.copy()
            __candidateFilters=__candidateFilters[__candidateFilters.polarity==self.polarity].reset_index(drop=True)
            if not(__candidateFilters.empty):
                self.__setNearScanParameters(self.currMS1ScanNum2tRmax,__candidateFilters,tR_thres=tR_thres)
            else:
                return                 

        if len(self.__nearMS2Scans)==0: return
        _nearestMS2Scan=self.__nearMS2Scans[self.__nearestMS2ScanIdx]

        if isinstance(ionsSearchList,type(None)):
            __ionsSearchList=self.getScan(_nearestMS2Scan)
        else:
            __ionsSearchList=ionsSearchList.copy()

        if  __ionsSearchList.empty: return
            
        self.MSfilter=self.lastMS2Filter.iloc[0]['filter']
        self.__currentCE=float(self.lastMS2Filter.iloc[0].CE.replace("hcd",""))

        __ionsSearchList=__ionsSearchList[__ionsSearchList.mz<=(self.__mzTargetFS+5)]
        if 'intensity' in __ionsSearchList.columns:
            __ionsSearchList=__ionsSearchList[__ionsSearchList.intensity<__xicMS1Template.intensity.max()]
            __ionsSearchList=__ionsSearchList[__ionsSearchList.intensity>=self.__intensityThres]
            if 'noise' in __ionsSearchList.columns:
                __ionsSearchList=__ionsSearchList[__ionsSearchList.intensity/__ionsSearchList.noise>self.__SN_thres].sort_values(['intensity'],ascending=False)

        else: __ionsSearchList['intensity']=None           
        __ionsSearchList=__ionsSearchList.reset_index(drop=True)
        dtR=((self.filterMaskLims[1]-self.filterMaskLims[0])/2.0)*(1.0-1.0/np.sqrt(6))
        maxRegVal=[self.filterMaskLims[0]+dtR,self.filterMaskLims[1]-dtR]


        processed_fragments=Parallel(n_jobs=self.__njobs)(delayed(assessFragments)(mz,
                                                                                   self.__MSfile,
                                                                                   self.__polarity,
                                                                                   self.__MSfilter,
                                                                                   self.__mzAcc,
                                                                                   self.__cmd,
                                                                                   self.filterMaskLims,
                                                                                   p_thres,
                                                                                   __xicMS1Template,
                                                                                   __ionsSearchList.intensity.iloc[j],
                                                                                   self.chromPeakMinNumOfPointsMS2,
                                                                                   isChi2Metric=self.fragmentationChi2Metric,
                                                                                   dInt=self.fragmentationXIC_dInt,maxReg=maxRegVal) for j,mz in enumerate(__ionsSearchList['mz']))
        

        k=0
        for fragment in processed_fragments:
            if not(fragment[0].empty):
                k+=1
                fragment[1]['CE']=self.__currentCE
                fragment[1]['fragmentNum']=k
                self.__probableFragmentIonsXICs=pd.concat([self.__probableFragmentIonsXICs,fragment[1]],sort=False,ignore_index=True)

                
        if not(self.__probableFragmentIonsXICs.empty):
            self.__probableFragmentIons=self.__probableFragmentIonsXICs[self.__probableFragmentIonsXICs.label=='ms2MaskedSection'].copy().reset_index(drop=True)
            tRc=self.__probableFragmentIons[self.__probableFragmentIons.fragmentNum==1].time[self.__probableFragmentIons[self.__probableFragmentIons.fragmentNum==1].intensity.idxmax()]
            self.__probableFragmentIons=self.__probableFragmentIons[self.__probableFragmentIons.time==tRc].reset_index(drop=True)
            self.__probableFragmentIons.intensity=self.__probableFragmentIons.intensity*self.__probableFragmentIons.intMax
            self.__probableFragmentIons=self.__probableFragmentIons[['mz','intensity','time','intMax','p']].rename(columns={'time':'tR','intMax':'maxInt','p':'p_MS1_MS2'})
            self.__probableFragmentIons['mzPrecursor']=self.mzTargetFS
            ms1XICSection=__xicMS1Template.copy()
            ms1XICSection=ms1XICSection[(ms1XICSection.time>=self.filterMaskLims[0]) &  (ms1XICSection.time<=self.filterMaskLims[1])]
            ms1XICSection['mz']=self.mzTargetFS
            ms1XICSection['label']="ms1Template"
            ms1XICSection['p']=""
            ms1XICSection['intMax']=""
            ms1XICSection['CE']=self.__currentCE
            ms1XICSection['fragmentNum']=0
            self.__probableFragmentIonsXICs=pd.concat([ms1XICSection,self.__probableFragmentIonsXICs],sort=False,ignore_index=True)
                

            
        if not(self.__probableFragmentIons.empty):
            self.__probableFragmentIons=self.__probableFragmentIons.sort_values(['mz']).reset_index(drop=True)
        self.MSfilter=__MSfilter
        return

    def __plotProbableFragmentsIonsXICs_panel(self, mz, mzIdx):

        __ions=self.probableFragmentIonsXICs.copy()[['label','mz','fragmentNum']].drop_duplicates()
        if isinstance(mz,float):
            ionIdx=__ions[__ions.mz==mz].fragmentNum.iloc[0]
        elif isinstance(mzIdx,int):
            ionIdx=mzIdx % len(self.__probableFragmentIons)
            mzIonIdx=self.probableFragmentIons.iloc[ionIdx].mz
            tmpDF=__ions[__ions.label=="ms2MaskedSection"]
            ionIdx=tmpDF[tmpDF.mz==mzIonIdx].fragmentNum.iloc[0]
        else:
            return

        plt.rcdefaults()
        plt.clf()
        fig = plt.figure()
        gs = fig.add_gridspec(5, hspace=0)
        axs = gs.subplots(sharex=True)
        __XIC=self.probableFragmentIonsXICs.copy()

        #Setting
        __XIC=__XIC[__XIC.label=="ms1Template"]
        mint=__XIC.time.min()
        maxt=__XIC.time.max()
        tRMax=__XIC.time.iloc[__XIC.intensity.idxmax()]
        xtextOffset=(__XIC.time.max()-tRMax)/2.0
        intMax=__XIC.intensity.max()

        ## FS XIC
        __XIC.intensity=__XIC.intensity/__XIC.intensity.max()        
        axs[0].plot(__XIC.time,__XIC.intensity,color='black',linewidth=2.5, alpha = 0.35)
        axs[1].plot(__XIC.time,__XIC.intensity,color='black',linewidth=0.85)
        _precMZ=float(__XIC.mz.iloc[0])
        axs[1].text(tRMax+xtextOffset,__XIC.intensity.max()*0.8,f"Int={float(intMax):.3E}")
        axs[1].legend([f"mz (ms1) ={_precMZ:.5F}"],loc=2)

        # FS sampled
        __XIC=self.probableFragmentIonsXICs.copy()
        __XIC=__XIC[__XIC.fragmentNum==ionIdx]
        __XIC=__XIC[__XIC.label==f"ms1SampledSection"]
        axs[0].plot(__XIC.time,__XIC.intensity,color='blue',linewidth=0.85)
        axs[2].plot(__XIC.time,__XIC.intensity,color='blue',linewidth=0.85)
        axs[0].text(tRMax+xtextOffset,__XIC.intensity.max()*0.4,f"tR={float(tRMax):.3F} min")
        axs[0].text(tRMax+xtextOffset,__XIC.intensity.max()*0.6,f"p={float(__XIC.p.iloc[0]):.2F}")
        axs[0].text(tRMax+xtextOffset,__XIC.intensity.max()*0.8,f"mz={float(__XIC.mz.iloc[0]):.5F}")
        axs[2].legend([f"ms1 (spl) ={_precMZ:.5F}"],loc=2)

        __XIC=self.probableFragmentIonsXICs.copy()
        __XIC=__XIC[__XIC.fragmentNum==ionIdx]
        __XIC=__XIC[__XIC.label==f"ms2MaskedSection"]
        axs[0].plot(__XIC.time,__XIC.intensity,color='red',linewidth=0.85)
        axs[3].plot(__XIC.time,__XIC.intensity,color='red',linewidth=0.85)
        axs[3].legend([f"mz (ms2)={float(__XIC.mz.iloc[0]):.5F}"],loc=2)
        axs[3].text(tRMax+xtextOffset,__XIC.intensity.max()*0.8,f"Int={__XIC.intMax.iloc[0]:.3E}")

        # FS and MS2 superposition
        __XIC=self.probableFragmentIonsXICs.copy()
        __XIC=__XIC[__XIC.fragmentNum==ionIdx]
        __XIC_ms1smpled=__XIC[__XIC.label==f"ms1SampledSection"]
        __XIC_ms2Masked=__XIC[__XIC.label==f"ms2MaskedSection"]
        axs[4].plot(__XIC_ms1smpled.time,__XIC_ms1smpled.intensity,color='blue',linewidth=0.85)
        axs[4].plot(__XIC_ms1smpled.time,__XIC_ms2Masked.intensity,color='red',linewidth=0.85)
        
        for i in range(0,5):
            axs[i].axvline(x=mint,ymin=0.0,ymax=1.0,color='red',linestyle = '-', alpha = 0.35,linewidth=0.9)
            axs[i].axvline(x=maxt,ymin=0.0,ymax=1.0,color='red',linestyle = '-', alpha = 0.35,linewidth=0.9)
            axs[i].axvline(x=tRMax,ymin=0.0,ymax=1.0,color='gray',linestyle = ':', alpha = 0.5,linewidth=0.85)
            axs[i].axvline(x=tRMax+0.1,ymin=0.0,ymax=1.0,color='gray',linestyle = ':', alpha = 0.5,linewidth=0.85)
            axs[i].axvline(x=tRMax-0.1,ymin=0.0,ymax=1.0,color='gray',linestyle = ':', alpha = 0.5,linewidth=0.85)
            axs[i].get_yaxis().set_visible(False)
            
        fig.supxlabel('time / min')
        plt.show()

    def __plotProbableFragmentsIonsXICs_stacked(self):

        __xics=self.probableFragmentIonsXICs.copy()
        __xics=__xics[(__xics.label.str.find('ms2MaskedSection')>=0) | (__xics.label.str.find('ms1Template')>=0)].sort_values(['mz','time'],ascending=False,ignore_index=True)

        plt.rcdefaults()
        plt.clf()

        for grp in __xics.groupby(['fragmentNum']):    
    
            __data=grp[1].copy().sort_values(['time'],ascending=True)
            if (__data.label.iloc[0]=="ms2MaskedSection") :
                __data=grp[1].copy()
                __data.intensity=__data.intensity*__data.intMax.iloc[0]
                plt.plot('time','intensity',data=__data,label='_Hidden',linewidth=0.85)
            else:
                plt.plot('time','intensity',data=__data,label=f"m/z={__data.mz.iloc[0]:.5F} (ms1)\nInt={__data.intensity.max():.2E}",linewidth=2.5)
    

        plt.title("XICs for matching fragment ions")
        plt.xlabel("time / min")
        plt.ylabel("Intensity")
        plt.legend(loc=2)
        plt.tight_layout()
        plt.show()
               
    
    def plotProbableFragmentIonsXICs(self,mz=None,mzIdx=None, areIonsStacked=True):

        if self.__probableFragmentIonsXICs.empty: return
        if areIonsStacked:
            self.__plotProbableFragmentsIonsXICs_stacked()
        else:
            self.__plotProbableFragmentsIonsXICs_panel(mz,mzIdx)
        


    def plotAnnotatedFragmentsIonsXICs(self,candidateIdx=0):

        if self.__probableFragmentIonsXICs.empty: return
        if self.__annotatedFragmentIons.empty: return

        inchikeys=self.__annotatedFragmentIons.inchikey.drop_duplicates().to_list()
        __mzs=self.__annotatedFragmentIons[self.__annotatedFragmentIons.inchikey==inchikeys[candidateIdx%len(inchikeys)]].mz.to_list()
        
        __xics=self.probableFragmentIonsXICs.copy().set_index('mz',drop=False)
        __xics=(__xics.loc[np.array(__mzs, dtype=__xics.index.dtype)])
        __xics=__xics[__xics.label=='ms2MaskedSection']
        __xics=pd.concat([self.probableFragmentIonsXICs[self.probableFragmentIonsXICs.label=="ms1Template"],__xics],sort=False,ignore_index=True).sort_values(['mz','time'],ascending=False).reset_index(drop=True)       
        __xics['ngp']=list(itertools.chain(*[ [idx]*len(grp[1]) for idx,grp in enumerate(__xics.groupby(['fragmentNum']))]))

        plt.rcdefaults()
        plt.clf()

        for idx,grp in enumerate(__xics.groupby(['ngp'])):       
            __data=grp[1].copy().sort_values(['time'],ascending=True)
            if (idx>0) :
                __data=grp[1].copy()
                __data.intensity=__data.intensity*__data.intMax.iloc[0]
                plt.plot('time','intensity',data=__data,label='_Hidden',linewidth=0.85)
            else:
                plt.plot('time','intensity',data=__data,label=f"m/z={__data.mz.iloc[0]:.5F} (ms1)\nInt={__data.intensity.max():.2E}",linewidth=2.5)
    

        plt.title(f"XICs for matching fragment ions: {inchikeys[candidateIdx%len(inchikeys)]}")
        plt.xlabel("time / min")
        plt.ylabel("Intensity")
        plt.legend(loc=2)
        plt.tight_layout()
        plt.show()




    def getNearestIonsFromScan(self,ionsList,scanNum,mzError_ppm=50):


        _nearestIons=self.getScan(scanNum).copy()
        _nearestIons=_nearestIons.iloc[[abs(_nearestIons.mz-mzIon).idxmin() for mzIon in ionsList.mz]]
        if _nearestIons.empty: return pd.DataFrame()
        _nearestIons.insert(2,'mzIon',ionsList.mz.to_list())
        _nearestIons.insert(3,'mzError_ppm',abs(_nearestIons.mz.to_numpy()-ionsList.mz.to_numpy())/ionsList.mz.to_numpy()*1E6)
        _nearestIons=_nearestIons[_nearestIons.mzError_ppm<=mzError_ppm]
        if scanNum in self.nearMS2Scans: _nearestIons['tR']=self.nearMS2Events.loc[scanNum-1].tR
        _nearestIons['scan']=scanNum

        return _nearestIons.reset_index(drop=True)


    def getNearestIonsFromFSScan(self,ionsList,mzError_ppm=50,tR_thres=None,ms1ScanNum=None,isPlotShown=False):
        __nearestIonsVault=pd.DataFrame()
        if ionsList.empty: return pd.DataFrame()
        __candidateFilters=self.filters.copy()      
        __candidateFilters=__candidateFilters[__candidateFilters.polarity==self.polarity].reset_index(drop=True)
        
        if not(__candidateFilters.empty):

            ms1ScanNum_=self.currMS1ScanNum2tRmax
            if isinstance(ms1ScanNum,int):
                ms1ScanNum_=ms1ScanNum

            __tR4Scan=self.eventsDF[self.eventsDF.nScan==ms1ScanNum_].tR.iloc[0]
            __maxInt=self.__currentFSXICPeak[self.__currentFSXICPeak.time==__tR4Scan].intensity.iloc[0]
            __xicMS1Template=self.__currentFSXICPeak[self.__currentFSXICPeak.intensity<=__maxInt].copy().reset_index(drop=True)
            tR_lims=tR_thres
            if isinstance(tR_thres,float): tR_lims=[__tR4Scan-tR_thres,__tR4Scan+tR_thres]
            self.setXICFilterMask(__xicMS1Template,tR_lims=tR_lims)                
            self.__setNearScanParameters(ms1ScanNum_,__candidateFilters,tR_thres)

            if len(self.nearMS2Scans)==0: return
            
            for scanNum in self.nearMS2Scans:
                __nearestIonsVault=pd.concat([__nearestIonsVault,
                                             self.getNearestIonsFromScan(ionsList,scanNum,mzError_ppm)],
                                            sort=False,ignore_index=True)

            if isPlotShown:
                plt.rcdefaults()
                plt.clf()
                tRMax=self.eventsDF[self.eventsDF.nScan==ms1ScanNum_].tR.iloc[0]
                __FSXIC=self.currentFSXICPeak.copy()
                __FSXIC=__FSXIC[(__FSXIC.time>=self.filterMaskLims[0]) &  (__FSXIC.time<=self.filterMaskLims[1])]
                __FSXIC.intensity=__FSXIC.intensity/__FSXIC.intensity.max()
                plt.plot('time','intensity',data=__FSXIC,label=f"mz (FS) {self.mzTargetFS:.5F}",alpha = 0.35,linewidth=0.85)
                for grp in __nearestIonsVault.groupby(['mzIon']):
                    __data=grp[1].copy()
                    __data.intensity=__data.intensity/__data.intensity.max()
                    plt.plot('tR','intensity',data=__data,label=f"mz (ms2) {__data.mzIon.iloc[0]:.5F}")
                plt.axvline(x=tRMax-0.1,ymin=0.0,ymax=1.0,color='gray',linestyle = ':', alpha = 0.5,linewidth=0.85)
                plt.axvline(x=tRMax,ymin=0.0,ymax=1.0,color='gray',linestyle = ':', alpha = 0.5,linewidth=0.85)
                plt.axvline(x=tRMax+0.1,ymin=0.0,ymax=1.0,color='gray',linestyle = ':', alpha = 0.5,linewidth=0.85)
                plt.axvline(x=self.filterMaskLims[0],ymin=0.0,ymax=1.0,color='red',linestyle = '-', alpha = 0.5,linewidth=0.85)
                plt.axvline(x=self.filterMaskLims[1],ymin=0.0,ymax=1.0,color='red',linestyle = '-', alpha = 0.5,linewidth=0.85)
                plt.legend(loc=2)
                plt.xlabel("time / min")
                plt.show()
                
                
        return __nearestIonsVault


    def getMultiPeaksInfo(self,numPeaks=None,tR_thres=0.1,k=1,isPlotShown=True,isPlotSaved=False,pltName="plot",title="",xicPeak=None):

        if isinstance(xicPeak,type(None)):
            __currentXICPeak=self.currentFSXICPeak.copy()
        else:
            __currentXICPeak=xicPeak.copy()
       
        if __currentXICPeak.empty: return pd.DataFrame()

        if isPlotShown:
            plt.clf()
            plt.rcdefaults()
            plt.plot(__currentXICPeak.time,__currentXICPeak.intensity,linewidth=0.85)
            plt.axhline(__currentXICPeak.intensity.mean(), linestyle = '--',color='red',linewidth=0.85)
            plt.axhline(__currentXICPeak.intensity.mean()+__currentXICPeak.intensity.std()*k,
                        linestyle = ':',color='red',linewidth=0.85)
            plt.title(title)
            plt.xlabel('time / min')
            plt.ylabel('Intensity')

               
        vLim=__currentXICPeak.intensity.mean()+__currentXICPeak.intensity.std()*k
        __currentXICPeak=__currentXICPeak[__currentXICPeak.intensity>=vLim]
        __currentXICPeak['relInt']=__currentXICPeak.intensity/__currentXICPeak.intensity.max()
        X=__currentXICPeak.time.to_numpy().reshape(-1,1)

        nPeaks=numPeaks

        if isinstance(nPeaks,type(None)):
            notOptimun=True
            nPeaks=1
            while notOptimun:
                nPeaks+=1
                if nPeaks>len(X):
                    break
                else:
                    find_clusters_kmeans=KMeans(nPeaks, random_state=42,n_init='auto')      
                    __currentXICPeak['peak']=find_clusters_kmeans.fit_predict(X)       
                    __currentXICPeakTmp=__currentXICPeak.loc[__currentXICPeak.groupby(['peak']).relInt.idxmax().to_list()].copy().drop(columns=['peak']).sort_values('time').reset_index(drop=True)
                    notOptimun=not(any((__currentXICPeakTmp.time[1:].to_numpy()-__currentXICPeakTmp.time[0:-1].to_numpy())<tR_thres))
            nPeaks=nPeaks-1

        find_clusters_kmeans=KMeans(nPeaks, random_state=42,n_init='auto')
        self.__multPeaksInfo=__currentXICPeak.copy()
        self.__multPeaksInfo['peak']=find_clusters_kmeans.fit_predict(X)       
        self.__multPeaksInfo=self.__multPeaksInfo.loc[self.__multPeaksInfo.groupby(['peak']).relInt.idxmax().to_list()].drop(columns=['peak']).sort_values('time').reset_index(drop=True)
        notOptimun=not(any((self.__multPeaksInfo.time[1:].to_numpy()-self.__multPeaksInfo.time[0:-1].to_numpy())<tR_thres))    
        __scans=self.eventsDF[['nScan','tR']].copy()
        self.__multPeaksInfo['nScan']=__scans.nScan.loc[[abs(__scans.tR-t).idxmin() for t in self.__multPeaksInfo.time]].to_list()


        if isPlotShown | isPlotSaved:
            for tR in self.__multPeaksInfo.time:
                plt.axvline(x=tR,ymin=0.0,ymax=1.0,color='gray',
                            linestyle = ':', alpha = 0.5,linewidth=0.85)
            if isPlotSaved:
                pltName_=pltName
                if ((pltName=="plot") & (title!="")): pltName_=''.join(filter(str.isalnum, title.strip().lower()))   
                plt.savefig(os.path.join(self.resultsPath,pltName_+".png"))

            plt.show()               
            plt.clf()
        
        return self.__multPeaksInfo


    def getNumberOfPointsOnMainPeak(self,tR_thres=0.1,ctR=-1.0):
        df=self.currentFSXICPeak.copy()
        if df.empty: return pd.DataFrame()
        tRmax=ctR
        if ctR<=0.0: tRmax=df.time.iloc[df.intensity.idxmax()]
        tRLims=[tRmax-tR_thres,tRmax+tR_thres]
        df=df[(df.time>=tRLims[0]) & (df.time<=tRLims[1])].reset_index(drop=True)
        return df[df.intensity>1E-6].reset_index(drop=True)


    def getFSEvents(self):

        __FSEvents=self.eventsDF.copy()
        __FSEvents=__FSEvents[__FSEvents.mzFilterEvent.str.find("ms2")<0].reset_index(drop=True)
        if not(__FSEvents.empty):
            __FSEvents['polarity']=__FSEvents.mzFilterEvent.apply(lambda x: x.split()[1])
            
        return __FSEvents


    def getCompleMassSpectrum(self):

        if self.__MSfile=="":
            return

        if os.path.exists(self.__completeMSFile): return

        if os.path.exists(self.__MSfile):
            __FSEvents=self.getFSEvents()
            startTime = time.monotonic()
            processed_scans= Parallel(n_jobs=self.__njobs)(delayed(getMSScan)(scan[1],self.__MSfile,self.__cmd) for scan in __FSEvents.iterrows())

            for scan in processed_scans:
                if not(scan.empty):

                    if (os.path.exists(self.__completeMSFile)):
                        scan.to_csv(self.__completeMSFile,sep="\t",index=False, mode='a',header=False)
                    else:
                        scan.to_csv(self.__completeMSFile,sep="\t",index=False)

            elapsed_time = time.monotonic() - startTime
            print(f"Elapsed time: {elapsed_time} using {self.__njobs} process")

            return 





        
