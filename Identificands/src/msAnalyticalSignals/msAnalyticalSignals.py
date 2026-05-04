# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import os
import subprocess
import io
import time,datetime
from IPython.display import display, clear_output
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from lcmsMeasurements import lcmsMeasurements
import Preprocessing.Normalizer as normalizer
from msMLModelsCommon import msMLModelsCommon
from msWaveletTransform import msWaveletTransform
msMLModelsComm=msMLModelsCommon()


def getFullyNormalizedSection(section):
    intensityNormMsSection=section.intensity/np.max(section.intensity)
    mzNormMsSection=section.mz-min(section.mz)
    invIntensityNormMsSection=1/np.log2(np.log2(np.log2(intensityNormMsSection*10E8)))
    normalizedSection=np.array([mzNormMsSection,intensityNormMsSection,invIntensityNormMsSection]).transpose()
    return normalizedSection

def isopatternInference(concernMSSection,intensityThreshold,writeInference=False):

    normalizedSection=getFullyNormalizedSection(concernMSSection)
    y_preds=msMLModelsComm.classifyIsotopicSignal(normalizedSection)
    isoPatternProbs=pd.DataFrame(y_preds)
    isoPatternProbs.columns = ["Spurious","M","M+1","M+2","M+3","M+4","M+5","M+6"] 

    if writeInference: concernMSSection.join(isoPatternProbs).join(pd.DataFrame({'ion':isoPatternProbs.idxmax(axis=1),
                                                                                 'p_ion':isoPatternProbs.max(axis=1)})).to_csv("./isoPatternSectionInference.tsv",sep="\t",index=False)

    isoPatternSection=concernMSSection.copy().reset_index(drop=True).join(pd.DataFrame({'ion':isoPatternProbs.idxmax(axis=1),
                                                                                 'p_ion':isoPatternProbs.max(axis=1)}))



    isoPatternSection=isoPatternSection[isoPatternSection.intensity>=intensityThreshold]
    isoPattern=isoPatternSection[isoPatternSection.ion!='Spurious'].reset_index(drop=True)  
    isoPattern=isoPattern.loc[isoPattern.groupby('ion')['p_ion'].idxmax().to_list()]

    if( np.any(np.unique(isoPattern['ion']==('M'))) & (len(isoPattern) > 1)):
        iSignals=list(isoPatternProbs.columns)[1:len(isoPattern)+1]
        iSignals=list(set(iSignals).intersection(set(isoPattern.ion.to_list())))
        isoPattern=isoPattern.set_index('ion',drop=False).loc[iSignals].reset_index(drop=True)

        if (len(isoPattern) > 1):
            isoPattern['patternCount']=0
            isoPattern['numOfIsoSignals']=len(isoPattern.ion)
            return isoPattern

    return pd.DataFrame()



def evalIsopatterns(isoPattern,intensityThreshold,writeInference=False):
    __labeledip=pd.DataFrame()

    if isoPattern.intensity.max()>=intensityThreshold:
        currentMSSection=isoPattern.copy().reset_index(drop=True)
        initialSize=len(currentMSSection)
        __labeledip=isopatternInference(currentMSSection,intensityThreshold,writeInference)

        currentMSSection=currentMSSection[currentMSSection.intensity<=currentMSSection.intensity[0]]
        if(initialSize!=len(currentMSSection)):
            __labeledip2=isopatternInference(currentMSSection,intensityThreshold,writeInference)
            if not(__labeledip2.empty):
                __labeledip=pd.concat([__labeledip,__labeledip2],sort=False,ignore_index=True)

    return __labeledip


class msAnalyticalSignals(lcmsMeasurements):

    def __init__(self):
        lcmsMeasurements.__init__(self)
        self.massSpectrumWT=msWaveletTransform()
        self.analyticalSectionNormalizer=normalizer.MsSectionsNormalization()
        self.__probableAnalyticalSignals=pd.DataFrame()
        self.__numOfisoPatternsFound=0
        self.__selectedPatterns=pd.DataFrame()
        self.__numOfSelectedIsotopicPatterns=-1
        self.__isoSignalAssessment_mzAcc=5.0
        self.__njobs=8

        #Current scan properties
        self.__currentScan=None
        self.__currentScan_tR=0.0
        self.__currentScanFilter=""
        self.__currentScanPolarity=""
        

        #Current cadidate attributes
        self.__currentCandidateIsoPattern=pd.DataFrame()
        self.__currentCandidateChromPeakAssessment=pd.DataFrame()
        self.__currentIsolatedIsotopicPattern=pd.DataFrame()
        self.__dbSearchMass=0.0

        #Current file properties
        self.__allCandidateAnalyticalSignals=pd.DataFrame()
        self.__rawFileProcessingtime=pd.DataFrame()
        self.__gpatterCounter=0

        #Analysis framework
        self.__signalsEventsInterval=[0,-1]

        ##Adquisition info
        self.__resultsPath="./resultsData"
        self.__bkresultsPath="./resultsData"
        self.__wtProbableAnalyticalSignalsLocationFile=""
        self.__completeWTtProbableAnalyticalSignalsLocationFile=""
        self.__allProbableAnalyticalSignals=pd.DataFrame()


    @property
    def currentScan(self):
        return self.__currentScan

    @currentScan.setter
    def currentScan(self,value):
        self.__currentScan=value
        if not(self.events): self.events=self.MSfile

        self.ms=self.getScan(self.__currentScan)
        events=self.eventsDF
        self.__currentScan_tR=events[events.nScan==self.__currentScan].tR.iloc[0]
        self.__currentScanPolarity=events[events.nScan==self.__currentScan].mzFilterEvent.iloc[0].split()[1]

        self.__currentScanFilter=' '.join(events[events.nScan==self.__currentScan].mzFilterEvent.iloc[0].split()[4:6])

        self.__currentCandidateIsoPattern=pd.DataFrame()
        self.__currentCandidateChromPeakAssessment=pd.DataFrame()


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
                if self.MSfile!="":
                    self.__resultsPath=os.path.join(self.__bkresultsPath,os.path.basename(self.MSfile).replace(".raw",""))
                    os.mkdir(self.__resultsPath)
        else:
            if not(os.path.exists(self.__bkresultsPath)):
                os.mkdir(self.__bkresultsPath)
                
            if self.MSfile!="":
                self.__resultsPath=os.path.join(self.__bkresultsPath,os.path.basename(self.MSfile).replace(".raw",""))
                if not(os.path.exists(self.__resultsPath)):
                    os.mkdir(self.__resultsPath)
            

        
    @property
    def njobs(self):
        return self.__njobs

    @njobs.setter
    def njobs(self,value):
        self.__njobs=value
        self._lcmsMeasurements__njobs=value


    @property
    def signalsEventsInterval(self):
        return self.__signalsEventsInterval

    @signalsEventsInterval.setter
    def signalsEventsInterval(self,value):
        self.__signalsEventsInterval=value

    @property
    def allWTProbableAnalyticalSignals(self):
        return self.__allProbableAnalyticalSignals
    
    @property
    def currentScan_tR(self):
        return self.__currentScan_tR

    @property
    def rawFileProcessingtime(self):
        return self.__rawFileProcessingtime

    @property
    def probableAnalyticalSignals(self):
        return self.__probableAnalyticalSignals


    @property
    def allCandidateAnalyticalSignals(self):
        return self.__allCandidateAnalyticalSignals

    @allCandidateAnalyticalSignals.setter
    def allCandidateAnalyticalSignals(self,value):
        self.__allCandidateAnalyticalSignals=value
    
    @property
    def numOfisoPatternsFound(self):
        return self.__numOfisoPatternsFound

    @property
    def selectedPatterns(self):
        return self.__selectedPatterns

    @property
    def currentIsolatedIsotopicPattern(self):
        return self.__currentIsolatedIsotopicPattern
    
    @property
    def currentCandidateIsoPattern(self):
        return self.__currentCandidateIsoPattern

    @currentCandidateIsoPattern.setter
    def currentCandidateIsoPattern(self,value,unitaryZ=True):

        if not(self.__selectedPatterns.empty):
            iterPatt=value
            iterPatt=iterPatt%(self.__selectedPatterns.patternCount.max()+1)
            if iterPatt==0: iterPatt=1

            self.__currentCandidateIsoPattern=self.__selectedPatterns[self.__selectedPatterns.patternCount==iterPatt][['mz','intensity','noise','p_ion','patternCount','charge']].copy().reset_index(drop=True)
            
            self.mzTarget=self.__currentCandidateIsoPattern.mz[0]

            if unitaryZ:
                self.currentCharge=1
            else:
                self.currentCharge=abs(self.__currentCandidateIsoPattern.charge.max())
                if self.currentCharge==0:self.currentCharge=1.0

            __ionSpecies="+H" 
            if self.__currentScanPolarity!='+':
                self.currentCharge=-1*abs(self.currentCharge)
                __ionSpecies="-H"
            self.setIonSpecies(__ionSpecies)
        return            


    def setCurrentCandidateIsoPattern(self,isoPattern,ionSpecies,dbSearchMass):

        self.__currentCandidateIsoPattern=isoPattern.copy()
        self.mzTarget=self.__currentCandidateIsoPattern.mz[0]
        self.__dbSearchMass=dbSearchMass
        self.accurateMass=self.__currentCandidateIsoPattern.mz[0]
        
        self.__currentScan=self.__currentCandidateIsoPattern['scan'].iloc[0]
        self.__currentScanPolarity=self.__currentCandidateIsoPattern.polarity[0]
        self.__currentScan_tR=self.__currentCandidateIsoPattern['tR'][0]
        self.__currentScanFilter=' '.join(self.eventsDF[self.eventsDF.nScan==self.__currentScan].mzFilterEvent.iloc[0].split()[4:6])

        self.polarity=self.__currentScanPolarity
        self.setIonSpecies(ionSpecies)
        self.currentCharge=self.currentIonSpecies.z.iloc[0]
        self.__currentCandidateChromPeakAssessment=pd.DataFrame()
        return            
    
    @property
    def isoSignalAssessment_mzAcc(self):
        return self.__isoSignalAssessment_mzAcc

    @isoSignalAssessment_mzAcc.setter
    def isoSignalAssessment_mzAcc(self,value):
        self.__isoSignalAssessment_mzAcc=value

    @property
    def dbSearchMass(self):
        return self.__dbSearchMass

    @dbSearchMass.setter
    def dbSearchMass(self,value):
        self.__dbSearchMass=value
    
    @property
    def currentCandidateChromPeakAssessment(self):
        return self.__currentCandidateChromPeakAssessment
        
       
    @property
    def candidateAnalyticalSignalsSummary(self):

        if not(self.__probableAnalyticalSignals.empty):
            return pd.DataFrame({'lastIsoSignal':["M+1","M+2","M+3","M+4","M+5","M+6","Total"],
              'Count':list(map(lambda nIon:Counter( (self.__probableAnalyticalSignals.ion=="M") & (self.__probableAnalyticalSignals['numOfIsoSignals']==nIon) ).get(True),range(2,8)))+
                                 [self.__numOfisoPatternsFound]
             })


    @property
    def numOfSelectedIsotopicPatterns(self):
        return self.__numOfSelectedIsotopicPatterns


    @property
    def completeWTtProbableAnalyticalSignalsLocationFile(self):
        return self.__completeWTtProbableAnalyticalSignalsLocationFile

    @completeWTtProbableAnalyticalSignalsLocationFile.setter
    def completeWTtProbableAnalyticalSignalsLocationFile(self,value):
        self.__completeWTtProbableAnalyticalSignalsLocationFile=value
        if value!="":
            self.__allProbableAnalyticalSignals=pd.read_csv(self.__completeWTtProbableAnalyticalSignalsLocationFile,sep="\t")


    def __setFilesNames(self,removeWTFile=False):
        self.__allCandidateASAFile=os.path.join(self.resultsPath,f"{os.path.basename(self.MSfile).replace('.raw','')}_nonTargetedScreeningCandidateAnalyticalSignals.tsv")
        if os.path.exists(self.__allCandidateASAFile): os.remove(self.__allCandidateASAFile)

        self.__ASAenlapsedTime=os.path.join(self.resultsPath,f"{os.path.basename(self.MSfile).replace('.raw','')}_nonTargetedScreeningCandidateAnalyticalSignals_enlapsedTime.tsv")
        if os.path.exists(self.__ASAenlapsedTime): os.remove(self.__ASAenlapsedTime)

        self.__wtProbableAnalyticalSignalsLocationFile=os.path.join(self.resultsPath,f"{os.path.basename(self.MSfile).replace('.raw','')}_nonTargetedScreeningWTProbableAnalyticalSignalsLocation.tsv")
        if removeWTFile:
            if os.path.exists(self.__wtProbableAnalyticalSignalsLocationFile): os.remove(self.__wtProbableAnalyticalSignalsLocationFile)

        self.__selectedIsoPatternsFile=os.path.join(self.resultsPath,f"{os.path.basename(self.MSfile).replace('.raw','')}_nonTargetedScreeningSelectedIsotopicPatterns.tsv")
        if os.path.exists(self.__selectedIsoPatternsFile): os.remove(self.__selectedIsoPatternsFile)



        
    def selectCandidateIsoPatterns(self,isEqualToIsoSignalsNum=False):

        if not(self.__probableAnalyticalSignals.empty):
            
            if self.numOfIsoSignals%7>1: 
                if isEqualToIsoSignalsNum: 
                    self.__selectedPatterns=self.__probableAnalyticalSignals[(self.__probableAnalyticalSignals['numOfIsoSignals']==self.numOfIsoSignals%7)].copy().reset_index(drop=True)
                else:
                    self.__selectedPatterns=self.__probableAnalyticalSignals[(self.__probableAnalyticalSignals['numOfIsoSignals']>=self.numOfIsoSignals%7)].copy().reset_index(drop=True)

                newIdx=dict(zip(self.__selectedPatterns.patternCount.unique(),range(1,len(self.__selectedPatterns.patternCount.unique())+1)))
                self.__selectedPatterns.patternCount=self.__selectedPatterns.patternCount.transform(lambda x:newIdx[x])                
            else:
                self.__selectedPatterns=self.__probableAnalyticalSignals.copy()     

            self.__numOfSelectedIsotopicPatterns=self.__selectedPatterns.patternCount.max()
            self.__selectedPatterns['scanNum']=self.currentScan
            if (os.path.exists(self.__selectedIsoPatternsFile)):
                self.__selectedPatterns.to_csv(self.__selectedIsoPatternsFile,sep="\t",index=False, mode='a',header=False)
            else:
                self.__selectedPatterns.to_csv(self.__selectedIsoPatternsFile,sep="\t",index=False)


   
            
    def getCandidateAnalyticalSignals(self):
        self.__probableAnalyticalSignals=pd.DataFrame()
        self.__foundPatterns=pd.DataFrame()
        if not(self.__currentScan is None):

            self.__numOfisoPatternsFound=0

            if self.__completeWTtProbableAnalyticalSignalsLocationFile=="":
                self.__probableAnalyticalSignals=self.massSpectrumWT.getProbablePatternsLocations(self.ms)
                self.__probableAnalyticalSignals['scanNum']=self.currentScan

                if self.__wtProbableAnalyticalSignalsLocationFile=="": self.__setFilesNames()
                if (os.path.exists(self.__wtProbableAnalyticalSignalsLocationFile)):
                    self.__probableAnalyticalSignals.to_csv(self.__wtProbableAnalyticalSignalsLocationFile,sep="\t",index=False, mode='a',header=False)
                else:
                    self.__probableAnalyticalSignals.to_csv(self.__wtProbableAnalyticalSignalsLocationFile,sep="\t",index=False)
            else:
                self.__probableAnalyticalSignals=self.__allProbableAnalyticalSignals[self.__allProbableAnalyticalSignals.scanNum==self.currentScan].copy()

                

            processed_signals = Parallel(n_jobs=self.__njobs)(delayed(evalIsopatterns)(grp[1],self.intensityThreshold) for grp in self.__probableAnalyticalSignals.groupby(['patternSectionIdx']))

            for el in processed_signals:
                if not(el.empty):
                    self.__gpatterCounter+=1
                    el.patternCount=self.__gpatterCounter
                    self.__foundPatterns=pd.concat([self.__foundPatterns,el],sort=False,ignore_index=True)
            
            if not(self.__foundPatterns.empty):
                self.__foundPatterns['StoN']=self.__foundPatterns.intensity/self.__foundPatterns.noise
                self.__foundPatterns['scan']=self.__currentScan
                self.__foundPatterns['tR']=self.__currentScan_tR
                self.__foundPatterns['polarity']=self.__currentScanPolarity
                self.__foundPatterns['scanSize']=len(self.ms)
                self.__probableAnalyticalSignals=self.__foundPatterns.copy()
                __idxs=(self.__probableAnalyticalSignals.groupby(["patternSectionIdx","ion"],as_index=False)[['p_ion']].idxmax()).p_ion.to_list()
                self.__probableAnalyticalSignals=self.__probableAnalyticalSignals.loc[__idxs]
                self.__probableAnalyticalSignals=self.__probableAnalyticalSignals.groupby(["patternSectionIdx","scanNum"],as_index=False).apply(lambda x: x.sort_values('ion')).reset_index(drop=True)
                
            else:
                self.__probableAnalyticalSignals=pd.DataFrame()


    def __searchAnalyticalSignalIntoScan(self,scanIdx,numOfIsoSignals=2):
            self.currentScan=scanIdx 
            self.getCandidateAnalyticalSignals()
            self.numOfIsoSignals=numOfIsoSignals
            self.selectCandidateIsoPatterns() 

    

    def getRawFileCandidateAnalyticalSignals(self,numOfIsoSignals=2,forceSearch=False):
        self.__rawFileProcessingtime=pd.DataFrame()
        self.__allCandidateAnalyticalSignals=pd.DataFrame()
        self.__gpatterCounter=0
        self.__setFilesNames(removeWTFile=forceSearch)
        __fsEvents=self.getFSEvents()
        hLim=min(self.__signalsEventsInterval[1],len(__fsEvents))
        if hLim==-1: hLim=len(__fsEvents)
        __fsEvents=__fsEvents.iloc[self.__signalsEventsInterval[0]:hLim]


        startTime = time.monotonic()
        pbarSpects = tqdm(total=len(__fsEvents), bar_format='{l_bar}{bar:100}{r_bar}{bar:-5b}')

        for idx,scan in enumerate(__fsEvents.nScan):

            scanStartTime = time.monotonic()
            clear_output(wait=False)
            os.system('clear')
            pbarSpects.set_description(f"Non-targeted processing analytical signals (njobs: {self.njobs}) scan index: {scan}")   
            self.__currentScan_tR=__fsEvents.tR.iloc[idx]
            self.__currentScanPolarity=__fsEvents.polarity.iloc[idx]
            self.__searchAnalyticalSignalIntoScan(scan,numOfIsoSignals)

            if not(self.probableAnalyticalSignals.empty):
                if os.path.exists(self.__allCandidateASAFile):
                    self.probableAnalyticalSignals.to_csv(self.__allCandidateASAFile,sep="\t",index=False, mode='a',header=False)
                else:
                    self.probableAnalyticalSignals.to_csv(self.__allCandidateASAFile,sep="\t",index=False)


            elapsed_time = time.monotonic() - scanStartTime
            self.__rawFileProcessingtime=pd.concat([self.__rawFileProcessingtime,
                                                pd.DataFrame({"scan":[scan],
                                                              "processTime":[elapsed_time]
                                                              })],sort=False,ignore_index=True)

            pbarSpects.update(1)
                
                
        clear_output(wait=True) 
        pbarSpects.bar_format
        pbarSpects.close()
        self.__allCandidateAnalyticalSignals=pd.read_csv(self.__allCandidateASAFile,sep="\t")
        self.__allCandidateAnalyticalSignals.patternCount=(self.__allCandidateAnalyticalSignals.ion=="M").apply(int).cumsum()
        self.__allCandidateAnalyticalSignals.to_csv(self.__allCandidateASAFile,sep="\t",index=False)
        self.__allCandidateAnalyticalSignals=pd.read_csv(self.__selectedIsoPatternsFile,sep="\t")
        self.__allCandidateAnalyticalSignals.patternCount=(self.__allCandidateAnalyticalSignals.ion=="M").apply(int).cumsum()
        self.__rawFileProcessingtime['totalTime']=time.monotonic() - startTime
        self.__rawFileProcessingtime.to_csv(self.__ASAenlapsedTime,sep="\t",index=False)
        
        if os.path.exists(self.__wtProbableAnalyticalSignalsLocationFile): os.remove(self.__wtProbableAnalyticalSignalsLocationFile) 
        self.__allProbableAnalyticalSignals=pd.DataFrame()
               
    def iterateCandidateIsoPattern(self,isPatternShown=False):

        if not(self.__currentCandidateIsoPattern.empty):
            iterCount=(self.__currentCandidateIsoPattern.patternCount[0]+1)%(self.__selectedPatterns.patternCount.max()+1)

            self.currentCandidateIsoPattern=iterCount

            if isPatternShown:
                self.plotCurrentCandidateIsoPattern()
              
            return self.currentCandidateIsoPattern


    def plotCurrentCandidateIsoPattern(self,figsize=(30,30)):

        if not(self.__currentCandidateIsoPattern.empty):
            plt.rcdefaults()
            fig = plt.figure()
            ax = fig.add_subplot()
            markerline, stemlines, baseline =ax.stem('mz','intensity',data=self.__currentCandidateIsoPattern,markerfmt='none')
            baseline.set_color('grey')
            baseline.set_linewidth(1)
            ax.set_xlabel('m/z')
            ax.set_ylabel('Intensity')
            ax.set_title(f"Extracted isotopic pattern: {self.__currentCandidateIsoPattern.patternCount[0]}")
            plt.tight_layout()
            plt.show()
            
    

    def assessChrom4CurrCandidateIsoPattern(self):

        self.xic=pd.DataFrame()
        self.currentFSXICPeak=pd.DataFrame()
        
        self.setAdqParameters(self.__currentScanFilter,
                              self.__currentScanPolarity,
                              self.__isoSignalAssessment_mzAcc,
                              self.currentCandidateIsoPattern.mz[0])

        self.mzTargetFS=self.currentCandidateIsoPattern.mz[0]
        xicHog=self.getXICArray()
        if not(isinstance(xicHog,tuple)):
            self.__currentCandidateChromPeakAssessment=pd.DataFrame()
            return
        
        
        self.xic=xicHog[1].copy()
        self.currentFSXICPeak=xicHog[1].copy()
        self.__currentCandidateChromPeakAssessment=msMLModelsComm.classifyChromatographicSignal(xicHog[0])
        tRmax=xicHog[1].time.iloc[xicHog[1].intensity.idxmax()]
        self.tR=tRmax
        self.currMS1ScanNum2tRmax=self.eventsDF[self.eventsDF.tR==tRmax].nScan.iloc[0]
        self.__currentCandidateChromPeakAssessment['tRmax']=tRmax
        self.__currentCandidateChromPeakAssessment['%B']=f"{self.getLCGradientOrganicComposition(tRmax):.1F}"

        self.__currentCandidateChromPeakAssessment['iMax2tRmax']=xicHog[1].intensity.max()
        self.__currentCandidateChromPeakAssessment['iMaxNoise']=self.__currentCandidateIsoPattern.noise.iloc[0]
        self.__currentCandidateChromPeakAssessment['iMaxCharge']=self.__currentCandidateIsoPattern.charge.iloc[0]
        self.__currentCandidateChromPeakAssessment['scanNum2tRmax']=self.currMS1ScanNum2tRmax
        self.__currentCandidateChromPeakAssessment['accurateMass']=self.accurateMass
                
        return 

    def getCurrCandidateIsoPatternSection(self,isSectionShown=False):
        self.__currentIsolatedIsotopicPattern=pd.DataFrame()
        if not(self.__currentCandidateIsoPattern.empty):

            section=self.ms[(self.ms.mz>=(self.__currentCandidateIsoPattern.mz.iloc[0]))].copy()           
            section=section[(section.mz<=(self.__currentCandidateIsoPattern.mz.iloc[-1]))]
            section['ionLabel']="S"
            section.loc[list(map(lambda i: abs(section.mz-self.__currentCandidateIsoPattern.mz.iloc[i]).idxmin(),range(len(self.__currentCandidateIsoPattern)))),'ionLabel']=["M"]+["M+"+str(i) for i in range(1,len(self.__currentCandidateIsoPattern))]
            
            self.isotopicPattUpperBound=section.mz.iloc[-1]
            self.accurateMass=section.mz.iloc[0]
            self.completeExpIsotopicPattSection=section.copy()
            self.__currentIsolatedIsotopicPattern=section[section.ionLabel!="S"].copy()
            self.__currentIsolatedIsotopicPattern['p_ion']=self.__currentCandidateIsoPattern.p_ion.to_list()
        
            if isSectionShown:
            

                plt.rcdefaults()
                plt.axvline(x = self.__currentCandidateIsoPattern.mz.iloc[0], color = 'g',linestyle=":",alpha=0.2)
                plt.axvline(x = self.__currentCandidateIsoPattern.mz.iloc[-1], color = 'g',linestyle=":",alpha=0.2)

                plt.stem(section.mz,
                         section.intensity,linefmt='b-',markerfmt='none',label="Señales alrededor del patron extraido")
                plt.stem(self.__currentCandidateIsoPattern.mz,
                         self.__currentCandidateIsoPattern.intensity,linefmt='y:',markerfmt='none',label="Patron isotopico extraido")
                
                plt.xlabel("m/z (Th)")
                plt.ylabel("Intensidad")
                plt.title(f"Seccion m/z completa para patron isotopico {self.__currentCandidateIsoPattern.patternCount[0]}")
                plt.legend()
                plt.tight_layout()
                plt.show()
        
            return section
        


    def getSignalFromWTProbableSignals(self,mzTarget,mzAcc=5.0):

        if not(self.__allProbableAnalyticalSignals.empty):
            dmz=mzAcc*mzTarget/1E6
            mzTarget_lower=mzTarget-dmz
            mzTarget_upper=mzTarget+dmz
            return self.__allProbableAnalyticalSignals[(self.__allProbableAnalyticalSignals.mz>=mzTarget_lower) & (self.__allProbableAnalyticalSignals.mz<=mzTarget_upper)]
