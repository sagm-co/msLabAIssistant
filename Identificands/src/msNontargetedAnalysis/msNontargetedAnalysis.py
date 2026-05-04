import sys,os,glob
from pathlib import Path
import time
from datetime import  datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain


from IPython.display import display, clear_output
from tqdm import tqdm


sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from msAnalyticalSignals import msAnalyticalSignals
from msChemicalFormulasInference import msChemicalFormulasInference
from msInteractionProduct import msInteractionProduct


class msNontargetedAnalysis(msAnalyticalSignals,msChemicalFormulasInference):
    
    def __init__(self):
        msAnalyticalSignals.__init__(self)
        msChemicalFormulasInference.__init__(self)
        self.__candidateMoleculesManager=msInteractionProduct()
        self.__detected=0

        ## Analysis framework
        self.__asAssesmentMinNPoints=2

        ##Current file info
        self.__rawFile=None
        self.__rawFileBaseName=""
        self.__fsEvents=[]

        
        ##Current frame analysis
        self.__currentScanIdx=0
        self.__currentIsotopicPatternIdx=-1
        self.__currentCandidateStructuralFormulas=pd.DataFrame()
        self.__selectedIsoPatternMetrics=""
        self.__interactionProdIdx=1
        self.__numOfdetectedCompounds=0
        self.__allCurrentProbableFragmentIons=pd.DataFrame()
        self.__allAnnotatedFragmentIons=pd.DataFrame()
        self.__candidateXICSignalMinNumOfPoints=5
        self.__monoisotIons=pd.DataFrame()
        self.__filteredCandidateIons=pd.DataFrame()
        self.isoSignalAssessment_mzAcc=5.0
        self.__removeFiles=False

        ## Detected info
        self.__allDetectedCandidateStructuralFormulas=pd.DataFrame()
        self.__identProcessingtime=pd.DataFrame()
        self.__fxicsFile=""

        ## Info flow control
        self.reloadAnnotationDataInfo=False

    def __iter__(self):
        return self

    def __next__(self):

        self.__currentIsotopicPatternIdx=(self.__currentIsotopicPatternIdx+1)%(self.numOfSelectedIsotopicPatterns+1)
        
        if self.__currentIsotopicPatternIdx==0:
            raise StopIteration
        else:
            if self.isVerbose:
                print(f"\nI.(msChemicalFormulasInference): \nScan:{self.__currentScanIdx}\nm/z: {self.mzTarget} - pattern {self.__currentIsotopicPatternIdx} from {self.numOfSelectedIsotopicPatterns}\n")
            self.searchCandidateStructures()
            
            return self.__currentCandidateStructuralFormulas


    @property
    def removeFiles(self):
        return self.__removeFiles

    @removeFiles.setter
    def removeFiles(self,value):
        self.__removeFiles=value

    
    @property
    def currentIsotopicPatternIdx(self):
        return self.__currentIsotopicPatternIdx

    
    @currentIsotopicPatternIdx.setter
    def currentIsotopicPatternIdx(self,value):
        self.__currentIsotopicPatternIdx=value

    @property
    def currentScanIdx(self):
        return self.__currentScanIdx

    @currentScanIdx.setter
    def currentScanIdx(self,value):
        self.__currentScanIdx=value
        self.__currentIsotopicPatternIdx=0
        
    @property
    def candidateXICSignalMinNumOfPoints(self):
        return self.__candidateXICSignalMinNumOfPoints

    @candidateXICSignalMinNumOfPoints.setter
    def candidateXICSignalMinNumOfPoints(self,value):
        self.__candidateXICSignalMinNumOfPoints=value

    
    @property
    def numOfdetectedCompounds(self):
        return self.__numOfdetectedCompounds

    @numOfdetectedCompounds.setter
    def numOfdetectedCompounds(self,value):
        self.__numOfdetectedCompounds=value

    @property
    def asAssesmentMinNPoints(self):
        return self.__asAssesmentMinNPoints
        
    @asAssesmentMinNPoints.setter
    def asAssesmentMinNPoints(self,value):
        self.__asAssesmentMinNPoints=value
    
    @property
    def allDetectedCandidateStructuralFormulas(self):
        return self.__allDetectedCandidateStructuralFormulas

    def cleanAllDetectedCandidateStructuralFormulas(self):
        self.__allDetectedCandidateStructuralFormulas=pd.DataFrame()
        self.__currentCandidateStructuralFormulas=pd.DataFrame()
    
    @property
    def interactionProducts(self):
        return self.__candidateMoleculesManager.interactionProducts

    @property
    def interactionReactants(self):
        return self.__candidateMoleculesManager.interactionReactants
   

    @interactionProducts.setter
    def interactionProducts(self,value):
        self.__candidateMoleculesManager.interactionProducts=value

    @property
    def selectedIsoPatternMetrics(self):
        return self.__selectedIsoPatternMetrics

    @selectedIsoPatternMetrics.setter
    def selectedIsoPatternMetrics(self,value):
        self.__selectedIsoPatternMetrics=value

    @property
    def interactionProdIdx(self):
        return self.__interactionProdIdx

    @interactionProdIdx.setter
    def interactionProdIdx(self,value):
        self.__interactionProdIdx=value
       
    @property
    def currentCandidateStructuralFormulas(self):
        return self.__currentCandidateStructuralFormulas

    @currentCandidateStructuralFormulas.setter
    def currentCandidateStructuralFormulas(self,value):
        self.__currentCandidateStructuralFormulas=value
    
    @property
    def currentCandidateMS2FragmentationAnnotation(self):
        return self.__candidateMoleculesManager.msFragmenter.msAnnotationData

    @currentCandidateMS2FragmentationAnnotation.setter
    def currentCandidateMS2FragmentationAnnotation(self,value):
        self.reloadAnnotationDataInfo=True
        self.__msAnnotationDataRL=value


    @property
    def fullScanEvents(self):

        if len(self.__fsEvents)==0:
            if self.MSfile!="":
                self.events=self.MSfile
                self.__fsEvents=self.eventsDF.copy()
                self.__fsEvents=self.__fsEvents.nScan[list(self.__fsEvents.mzFilterEvent.str.find("Full ms ")>0)].to_list()

        return self.__fsEvents
               
    @property
    def allCurrentDetectedProbableFragmentIons(self):
        return self.__allCurrentProbableFragmentIons

    @property
    def allAnnotatedFragmentIons(self):
        return self.__allAnnotatedFragmentIons

    @allAnnotatedFragmentIons.setter
    def allAnnotatedFragmentIons(self,value):
        self.__allAnnotatedFragmentIons=value

    
    @property
    def rawFile(self):
        return self.__rawFile


    @rawFile.setter
    def rawFile(self,value):
        self.__allDetectedCandidateStructuralFormulas=pd.DataFrame()
        self.__numOfdetectedCompounds=0
        self.__rawFile=str(Path(value).resolve())
        self.MSfile=self.__rawFile
        self.events=self.__rawFile
        self.__rawFileBaseName=os.path.basename(self.__rawFile).replace('.raw','')
        self.resultsPath=self.resultsPath

        if self.currentIonSpeciesList.empty:
            self.currentIonSpeciesList=self.commonIonSpecies[['ionSpecies','z_apported']].rename(columns={'ionSpecies':'formula','z_apported':'z'})

        #WT
        self.__setWTFile()
            
        #FS XICs
        self.__nontargetedXICsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningFSXICs.tsv")
        if os.path.exists(self.__nontargetedXICsFile) & self.__removeFiles: os.remove(self.__nontargetedXICsFile)

        self.__nontargetedIsotopicPatternsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningIsotopicPatterns.tsv")
        if os.path.exists(self.__nontargetedIsotopicPatternsFile) & self.__removeFiles: os.remove(self.__nontargetedIsotopicPatternsFile)
                        
        #Compound detection info
        self.__ionsDetectionFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningAllDetectedCompounds.tsv")
        if os.path.exists(self.__ionsDetectionFile) & self.__removeFiles: os.remove(self.__ionsDetectionFile)
            
        #Fragments data
        self.__fxicsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningXICs4fragmentsIonsFound.tsv")
        if os.path.exists(self.__fxicsFile) & self.__removeFiles: os.remove(self.__fxicsFile)

        self.__fragmentIonsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningFragmentIonsFound.tsv")
        if os.path.exists(self.__fragmentIonsFile) & self.__removeFiles: os.remove(self.__fragmentIonsFile)

        self.__fragmentIonsAnnotationsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningFragmentsIonsFoundAnnotated.tsv")
        if os.path.exists(self.__fragmentIonsAnnotationsFile) & self.__removeFiles: os.remove(self.__fragmentIonsAnnotationsFile)
        
        #enlapsed time
        self.__enlapsedTimeFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningIdentification_enlapsedTime.tsv")
        if os.path.exists(self.__enlapsedTimeFile) & self.__removeFiles: os.remove(self.__enlapsedTimeFile)
        
        #ions data
        self.__monoIsotopicAssesmentsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningMonoIsotopicalIonsAssessment.tsv")
        self.__allIonsAssessmentsFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningFullCandidateIonsAssessment.tsv")
        self.__completeChromChemSpaceFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningCandidatesIntoCompleteChromChemSpace.tsv")
        self.__filteredIonsAssessmentFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningFilteredCandidateIonsAssessment.tsv")
        

        #Analytical signals assessment
        self.__allCandidateASAFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningCandidateAnalyticalSignals.tsv")
        self.__selectedCandidateASAFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningSelectedIsotopicPatterns.tsv")
        

    @property
    def commonIonSpecies(self):
        return self.__candidateMoleculesManager.commonIonSpecies

    @commonIonSpecies.setter
    def commonIonSpecies(self,file):
        self.__candidateMoleculesManager.commonIonSpecies=pd.read_csv(file,sep="\t")

    @property
    def identificationProcessingtime(self):
        return self.__identProcessingtime        
        
    @property
    def monoisotIons(self):
        return self.__monoisotIons
    
    @property
    def filteredCandidateIons(self):
        return self.__filteredCandidateIons

    def __setWTFile(self):
        self.__completeWTtProbableAnalyticalSignalsLocationFile=os.path.join(self.resultsPath,f"{self.__rawFileBaseName}_nonTargetedScreeningWTProbableAnalyticalSignalsLocation.tsv")
        self.completeWTtProbableAnalyticalSignalsLocationFile=""
        if os.path.exists(self.__completeWTtProbableAnalyticalSignalsLocationFile):
            self.completeWTtProbableAnalyticalSignalsLocationFile=self.__completeWTtProbableAnalyticalSignalsLocationFile

    
    def __searchStructuralFormulasInDB(self,projErrors):

        if not(projErrors.empty):
            for idx,candidateMolecularFormula in enumerate(projErrors.interactionProductFormula):

                self.__candidateMoleculesManager.formula=(candidateMolecularFormula,self.currentCharge)
                candidateFreeMolecule=self.__candidateMoleculesManager.transformFormulaInverse(*list(self.currentIonSpecies[['formula','z']].iloc[0]))

                self.dbStructuralFormulasSearch(candidateFreeMolecule)
                if not(self.candidateStructuralFormulas.empty):
                    
                    tmpDF=self.currentCandidateChromPeakAssessment.copy()
                    tmpDF['massError_ppm']=projErrors.dExactMass_ppm.iloc[idx]
                    tmpDF['interactionProductCharge']=self.currentCharge
                    tmpDF['interactionProductFormula']=candidateMolecularFormula
                    tmpDF['isoPatternIdx']=self.currentIsotopicPatternIdx
                    tmpDF=self.candidateStructuralFormulas.join(tmpDF,how="cross")
                    tmpDF.index=[projErrors.iloc[idx].rankSource]*len(tmpDF)
                    tmpDF=self.__appendErrorProjections(tmpDF,projErrors.iloc[idx].rankSource)
                    tmpDF=tmpDF.reset_index(drop=False).rename(columns={"index": "rankSource"})
                    self.__numOfdetectedCompounds+=1
                    tmpDF.insert(0,'CompNum',[self.__numOfdetectedCompounds]*len(tmpDF))
                    self.__currentCandidateStructuralFormulas=pd.concat([self.__currentCandidateStructuralFormulas,
                                                                         tmpDF],sort=False,ignore_index=True)

            self.__currentCandidateStructuralFormulas=self.__currentCandidateStructuralFormulas.reset_index(drop=True)

        return
            
    def __isopatternsAssessment(self):

        self.assessIsoPatternByIsoPatternsProjection()
        projErrors=self.patternsVecProjectionErrors
        self.selectedIsoPatternMetrics="vectorialProjection"

        return projErrors

    def __candidateStructureInferences(self):

        if self.__currentCandidateStructuralFormulas.empty: return
        self.__currentCandidateStructuralFormulas['p_amenability']=0.0
        self.__currentCandidateStructuralFormulas['tR_inference']=0.0
        self.__currentCandidateStructuralFormulas['U_tR_inference']=0.0
        self.__currentCandidateStructuralFormulas['p_tR_inference']=0.0
        self.__currentCandidateStructuralFormulas['p_ionizedSpecies']=0.0

        for idx,candidate in enumerate(self.__currentCandidateStructuralFormulas.iterrows()):
            self.__candidateMoleculesManager.smiles=candidate[1].canonicalSmiles
            if self.__candidateMoleculesManager.smiles!=None:

                __tRInference=self.__candidateMoleculesManager.getRetentionTimeInference(self.tR)
                if not(__tRInference.empty):
                    self.__currentCandidateStructuralFormulas.at[idx,'tR_inference']=__tRInference['tR'].iloc[0]
                    self.__currentCandidateStructuralFormulas.at[idx,'U_tR_inference']=__tRInference['U_tR'].iloc[0]
                    self.__currentCandidateStructuralFormulas.at[idx,'p_tR_inference']=__tRInference['p_tR'].iloc[0]
                    self.__currentCandidateStructuralFormulas.at[idx,'p_amenability']=self.__candidateMoleculesManager.getAmenabilityInference()['p'].iloc[0]
                    __pIonInteraction=self.__candidateMoleculesManager.getIonizationSpeciesPdPInference()
                    __pIonInteraction=__pIonInteraction[__pIonInteraction.ionSpecies==self.__currentCandidateStructuralFormulas['ionizedSpecies'].iloc[0]]['p'].iloc[0]                
                    self.__currentCandidateStructuralFormulas.at[idx,'p_ionizedSpecies']=__pIonInteraction
                    
                
        if self.reloadAnnotationDataInfo: self.__candidateMoleculesManager.msFragmenter.msAnnotationData=self.__msAnnotationDataRL


    def __mzBasedSearch(self,ionNum):
        self.getStructuralFormulasFromDBMassSearch(self)

        if not(self.candidateStructuralFormulas.empty):
            self.getIsoPatternsForCandidateMolFormulas()                      
            tmpDF=self.currentCandidateChromPeakAssessment.copy()

            __xic=self.xic.copy()
            __xic['ionNum']=ionNum
            if (os.path.exists(self.__nontargetedXICsFile)):
                __xic.to_csv(self.__nontargetedXICsFile,sep="\t",index=False, mode='a',header=False)
            else:
                __xic.to_csv(self.__nontargetedXICsFile,sep="\t",index=False)
            del __xic

            __isoPatt=self.isoPatternToAssess.copy()
            __isoPatt['ionNum']=ionNum
            if (os.path.exists(self.__nontargetedIsotopicPatternsFile)):
                __isoPatt.to_csv(self.__nontargetedIsotopicPatternsFile,sep="\t",index=False, mode='a',header=False)
            else:
                __isoPatt.to_csv(self.__nontargetedIsotopicPatternsFile,sep="\t",index=False)

            

            self.__currentCandidateStructuralFormulas=self.candidateStructuralFormulas.copy()
            self.__currentCandidateStructuralFormulas=self.__currentCandidateStructuralFormulas.drop(columns=['ionizedSpecies'])
            self.__numOfdetectedCompounds+=1
            self.__currentCandidateStructuralFormulas.insert(0,'isoPatternIdx',[self.currentIsotopicPatternIdx]*len(self.__currentCandidateStructuralFormulas))
            self.__currentCandidateStructuralFormulas.insert(1,'scanNum2tRmax',[tmpDF['scanNum2tRmax'].iloc[0]]*len(self.__currentCandidateStructuralFormulas))
            self.__currentCandidateStructuralFormulas=self.__currentCandidateStructuralFormulas.reset_index(drop=False).rename(columns={"index": "rankSource"})
            self.__currentCandidateStructuralFormulas.insert(0,'CompNum',[self.__numOfdetectedCompounds]*len(self.__currentCandidateStructuralFormulas))
            self.__currentCandidateStructuralFormulas.insert(1,'ionNum',[ionNum]*len(self.__currentCandidateStructuralFormulas))
            self.__currentCandidateStructuralFormulas['polarity']=self.polarity
            self.__currentCandidateStructuralFormulas['iMax2tRmax']=tmpDF['iMax2tRmax'].iloc[0]
            self.__currentCandidateStructuralFormulas['iMaxNoise']=tmpDF['iMaxNoise'].iloc[0]
            self.__currentCandidateStructuralFormulas['iMaxCharge']=tmpDF['iMaxCharge'].iloc[0]
            self.__currentCandidateStructuralFormulas['identificationApproach']='non-targeted-screening'
            self.__currentCandidateStructuralFormulas['interactionProductAccurateMass']=tmpDF['accurateMass'].iloc[0]            
            self.__currentCandidateStructuralFormulas['massError_ppm']=(self.__currentCandidateStructuralFormulas.searchMZ-self.__currentCandidateStructuralFormulas.bmonoisotopicMass)/self.__currentCandidateStructuralFormulas.bmonoisotopicMass*1E6
            self.__currentCandidateStructuralFormulas['ionizedSpecies']=self.candidateStructuralFormulas.copy()['ionizedSpecies']
            self.__currentCandidateStructuralFormulas['interactionProductCharge']=self.currentCharge
            self.__currentCandidateStructuralFormulas['interactionProductFormula']=self.candidateInteractionProductFormulas.Formula.to_list()
            self.__currentCandidateStructuralFormulas['einteractionProductFormula']=[ tpl[0]+tpl[1] for tpl in zip(self.__currentCandidateStructuralFormulas['molecularFormula'].to_list(), self.__currentCandidateStructuralFormulas['ionizedSpecies'].to_list())]
            self.__currentCandidateStructuralFormulas=self.__currentCandidateStructuralFormulas.set_index('interactionProductFormula',drop=False).join(self.currentCandidateFormulas[['Formula','interactionProductExactMass']].set_index('Formula'),sort=False).reset_index(drop=True).sort_values(['rankSource']).reset_index(drop=True)
            self.__currentCandidateStructuralFormulas['tRmax']=tmpDF['tRmax'].iloc[0]
            self.__currentCandidateStructuralFormulas['%B']=tmpDF['%B'].iloc[0]
            self.__isopatternsAssessment() 
            __projErrors=self.patternsVecProjectionErrors.iloc[:,[1,2,3,4,6]].copy().sort_values(['rankSource'])           
            self.__currentCandidateStructuralFormulas['vectorialProj_intensityError']=__projErrors['vectorialProj_intensityError']
            self.__currentCandidateStructuralFormulas['vectorialProj_mzError']=__projErrors['vectorialProj_mzError']
            self.__currentCandidateStructuralFormulas['vectorialProj_combinedError']=__projErrors['vectorialProj_combinedError']
            self.__currentCandidateStructuralFormulas['p_noise']=tmpDF['p_noise'].iloc[0]
            self.__currentCandidateStructuralFormulas['p_chromProblems']=tmpDF['p_chromProblems'].iloc[0]
            self.__currentCandidateStructuralFormulas['p_acceptableSignal']=tmpDF['p_acceptableSignal'].iloc[0]
            self.__currentCandidateStructuralFormulas['qltySignal']=self.__currentCandidateStructuralFormulas.iloc[0,29:32].idxmax()=='p_acceptableSignal'
            self.__currentCandidateStructuralFormulas['p_vectorialProj']=__projErrors['p_vectorialProj']
            self.__currentCandidateStructuralFormulas=self.__currentCandidateStructuralFormulas.drop(columns=['searchMZ'])
            

            

    def __fragmentsIonsAssessment(self):
        if(self.probableFragmentIons.empty): return
        __ms4annotation=self.probableFragmentIons[['mz','intensity']].copy()
        self.__candidateMoleculesManager.msFragmenter.msSpectrum4Annotation=__ms4annotation
        __extDB=self.candidateStructuralFormulas[['bmonoisotopicMass','IDkey','molecularFormula','canonicalSmiles','inchikey']].copy()
        __extDB=__extDB.rename(columns={'bmonoisotopicMass':'MonoisotopicMass','IDkey':'Identifier','molecularFormula':'MolecularFormula','canonicalSmiles':'SMILES','inchikey':'InChI'})
        __db_name=f"db_d{self.__numOfdetectedCompounds}_{datetime.now().strftime('%y%m%d%H%M%S%f')}"
        __extDB.to_csv(__db_name,sep=",",index=False)
        __candidate=self.candidateStructuralFormulas.iloc[0]
        self.__candidateMoleculesManager.msFragmenter.annotateSpectrum(__candidate.canonicalSmiles,
                                                                       __candidate.ionizedSpecies,
                                                                       __candidate.molecularFormula,
                                                                       __db_name,
                                                                       __extDB.MonoisotopicMass.iloc[0]
                                                                       )
        os.remove(__db_name)

        
    def __saveDectectionInfo(self,ionNum):

        if self.probableFragmentIons.empty: return
        tmpDF=self.probableFragmentIonsXICs.copy()
        tmpDF.insert(0,'CompNum',[self.__numOfdetectedCompounds]*len(tmpDF))
        tmpDF.insert(1,'ionNum',[ionNum]*len(tmpDF))
        
        if (os.path.exists(self.__fxicsFile) ):
            tmpDF.to_csv(self.__fxicsFile,sep="\t",index=False, mode='a',header=False)
        else:
            tmpDF.to_csv(self.__fxicsFile,sep="\t",index=False)

            
        #Save fragment ions
        self.__allCurrentProbableFragmentIons=self.probableFragmentIons.copy()
        self.__allCurrentProbableFragmentIons.insert(0,'CompNum',[self.__numOfdetectedCompounds]*len(self.__allCurrentProbableFragmentIons))
        self.__allCurrentProbableFragmentIons.insert(1,'ionNum',[ionNum]*len(self.__allCurrentProbableFragmentIons))
        
        if (os.path.exists(self.__fragmentIonsFile)):
            self.__allCurrentProbableFragmentIons.to_csv(self.__fragmentIonsFile,sep="\t",index=False, mode='a',header=False)
        else:
            self.__allCurrentProbableFragmentIons.to_csv(self.__fragmentIonsFile,sep="\t",index=False)
        
        if not(self.__candidateMoleculesManager.msFragmenter.msAnnotationData.empty):
           
            self.__allAnnotatedFragmentIons=self.__candidateMoleculesManager.msFragmenter.msAnnotationData.copy()
            self.__allAnnotatedFragmentIons=self.__allAnnotatedFragmentIons.astype({'mz':'float64'}).set_index('mz').join(self.probableFragmentIons[['mz','p_MS1_MS2','mzPrecursor']].copy().set_index('mz')).sort_values(['Score','inchikey'],ascending=False).reset_index() 
            self.__allAnnotatedFragmentIons.insert(0,'CompNum',[self.__numOfdetectedCompounds]*len(self.__allAnnotatedFragmentIons))
            self.__allAnnotatedFragmentIons.insert(1,'ionNum',[ionNum]*len(self.__allAnnotatedFragmentIons))
            self.annotatedFragmentIons=self.__allAnnotatedFragmentIons
            if (os.path.exists(self.__fragmentIonsAnnotationsFile)):
                self.__allAnnotatedFragmentIons.to_csv(self.__fragmentIonsAnnotationsFile,sep="\t",index=False, mode='a',header=False)
            else:
                self.__allAnnotatedFragmentIons.to_csv(self.__fragmentIonsAnnotationsFile,sep="\t",index=False)

            #It adds score by ms2 metfrag's annotation
            __scores=self.__allAnnotatedFragmentIons[['IDKey','Score']].drop_duplicates().rename(columns={'Score':'p_ms2','IDKey':'idx'})
            self.__currentCandidateStructuralFormulas['idx']=self.__currentCandidateStructuralFormulas.copy().IDkey
            self.__currentCandidateStructuralFormulas=self.__currentCandidateStructuralFormulas.set_index('idx').join(__scores.set_index('idx')).reset_index(drop=True)

            #It adds quality score for ms2 fragments with just  metfrag's annotation
            __ms2ProfileQuality=self.__allAnnotatedFragmentIons[['IDKey']].copy().drop_duplicates()
            __ms2ProfileQuality['numMS2Fragments']=[len(grp[1]) for grp in self.__allAnnotatedFragmentIons.groupby(['IDKey'],sort=False)]
            __ms2ProfileQuality['p_ms2Qlty']=[self.assessMS2ProfileQuality(grp[1]) for grp in self.__allAnnotatedFragmentIons.groupby(['IDKey'],sort=False)]
            self.__currentCandidateStructuralFormulas=self.__currentCandidateStructuralFormulas.set_index('IDkey',drop=False).join(__ms2ProfileQuality.set_index('IDKey')).reset_index(drop=True)


        else:
            self.__currentCandidateStructuralFormulas['p_ms2']=np.nan
            self.__currentCandidateStructuralFormulas['numMS2Fragments']=len(self.probableFragmentIons)
            self.__currentCandidateStructuralFormulas['p_ms2Qlty']=self.assessMS2ProfileQuality(self.probableFragmentIons)

            
        
        

    def showCurrentCandidateSFormula(self,formulaIdx=0):

        if not(self.__currentCandidateStructuralFormulas.empty):
            formulaIdx=formulaIdx%len(self.__currentCandidateStructuralFormulas)
            __extSmiles=self.__currentCandidateStructuralFormulas.iloc[formulaIdx].canonicalSmiles
            __name=self.__currentCandidateStructuralFormulas.iloc[formulaIdx]['name']
            return self.__candidateMoleculesManager.showMolecularStructure(__extSmiles,__name)
        

    def searchAnalyticalSignalIntoScan(self,numOfIsoSignals=2):

        if  ( self.probableAnalyticalSignals.empty | ( self.currentScan != self.__currentScanIdx) ):

            self.currentScan=self.__currentScanIdx
            self.getCandidateAnalyticalSignals()            
            self.numOfIsoSignals=numOfIsoSignals
            self.selectCandidateIsoPatterns() 

          
    def searchCandidateStructures(self,assessFragmentIons=False):
        self.__candidateMoleculesManager.msFragmenter.resetFragmentsData()
        self.searchAnalyticalSignalIntoScan()
        if self.probableAnalyticalSignals.empty: return
        
        startTime = time.monotonic()
        self.currentCandidateIsoPattern=self.__currentIsotopicPatternIdx  
        self.getCurrCandidateIsoPatternSection()
        self.__mzBasedSearch()

        if not(self.__currentCandidateStructuralFormulas.empty):
            self.__candidateStructureInferences()
            self.searchNontargetedProbableFragmentIons()
            if assessFragmentIons: self.__fragmentsIonsAssessment()
            if not(self.probableFragmentIons.empty): self.__saveDectectionInfo()

            
        enlapsed_time = time.monotonic() - startTime
        print(f"Pattern and structural formula search time: {enlapsed_time} s")
          
        return 

    
    def plotCurrentIsoPatternSection(self):

        if ( (not(self.completeExpIsotopicPattSection.empty)) & (not(self.currentCandidateIsoPattern.empty)) ):
            sSignals=self.completeExpIsotopicPattSection[self.completeExpIsotopicPattSection.ionLabel=="S"]
            
            plt.rcdefaults()
            plt.clf()
            plt.plot(sSignals.mz,sSignals.intensity,'bo',alpha=0.3)
            plt.plot(self.currentCandidateIsoPattern.mz,self.currentCandidateIsoPattern.intensity,'go',alpha=0.3,label="Experimental pattern")
            plt.stem(self.currentCandidateIsoPattern.mz,self.currentCandidateIsoPattern.intensity,linefmt='g-',markerfmt='none',basefmt="none")
            plt.ylim(0.0,self.completeExpIsotopicPattSection.intensity.max()*1.05)
            plt.xlabel('m/z Th')
            plt.ylabel('Intensity')
            
            plt.title(f"m/z: {round(self.mzTarget,5)} Th",fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.clf()


    def plotCandidateCompoundXIC(self,candidateIdx=0):

        if not(self.currentCandidateStructuralFormulas.empty):
            __candidateInfo=self.currentCandidateStructuralFormulas.iloc[candidateIdx%len(self.currentCandidateStructuralFormulas)]
            dtR=__candidateInfo.tR_inference-self.tR
            self.plotXIC(plot2dtR=dtR,U_tR=__candidateInfo.U_tR_inference,title=__candidateInfo['name'])
            

    def __loadCandidateSignals(self,file):
        self.allCandidateAnalyticalSignals=pd.read_csv(file,sep="\t")
        self.allCandidateAnalyticalSignals.patternCount=(self.allCandidateAnalyticalSignals.ion=="M").apply(int).cumsum()


    def __getCompoundsbyMZcluster(self,ionNum):
        __candidateXIC=self.__monoisotIons.copy().sort_values(['tR'])[['tR']].drop_duplicates() 
        __candidateXIC=__candidateXIC.set_index('tR').join(self.__monoisotIons[self.__monoisotIons.massGroup==ionNum].copy()[['tR','intensity']].set_index('tR')).fillna(0.0).reset_index() 
        return 

        
        
    def assessCandidateAnalyticalSignals(self,candidateSignalsFile=None,dmzThresClustering=2.0,tRThresClustering=0.1,polarity="+",ionFFreq=0.1,forceSearch=False): 

        if not(forceSearch):
            if os.path.exists(self.__filteredIonsAssessmentFile):
                self.__filteredCandidateIons=pd.read_csv(self.__filteredIonsAssessmentFile,sep="\t")
                if os.path.exists(self.__selectedCandidateASAFile):
                    self.__loadCandidateSignals(self.__selectedCandidateASAFile)
                elif os.path.exists(self.__allCandidateASAFile):
                    self.__loadCandidateSignals(self.__allCandidateASAFile)
                return
            
        
        if os.path.exists(self.__monoIsotopicAssesmentsFile): os.remove(self.__monoIsotopicAssesmentsFile)
        if os.path.exists(self.__allIonsAssessmentsFile): os.remove(self.__allIonsAssessmentsFile)
        if os.path.exists(self.__filteredIonsAssessmentFile): os.remove(self.__filteredIonsAssessmentFile)
        if os.path.exists(self.__completeChromChemSpaceFile): os.remove(self.__completeChromChemSpaceFile)
        
        self.__setWTFile()

        if (candidateSignalsFile==None) & (self.allCandidateAnalyticalSignals.empty):

            if os.path.exists(self.__selectedCandidateASAFile):
                self.__loadCandidateSignals(self.__selectedCandidateASAFile)
            elif os.path.exists(self.__allCandidateASAFile):
                self.__loadCandidateSignals(self.__allCandidateASAFile)
            else:
                print('I.(msNontargetedAnalysis): Starting analytical signals search')
                self.getRawFileCandidateAnalyticalSignals(forceSearch=forceSearch)
        elif candidateSignalsFile!=None:
            self.__loadCandidateSignals(candidateSignalsFile)
        elif (candidateSignalsFile==None) & (not(self.allCandidateAnalyticalSignals.empty)):
            self.__loadCandidateSignals(self.__selectedCandidateASAFile)

        ## Clustering by m/z
        self.__monoisotIons=self.allCandidateAnalyticalSignals[self.allCandidateAnalyticalSignals.ion=="M"].copy().sort_values(['mz'],ascending=True).reset_index(drop=False).rename(columns={'index':'orgIndex'})
        self.__monoisotIons=self.__monoisotIons[self.__monoisotIons.intensity>=self.intensityThreshold]

        self.__monoisotIons['dmz_ppm']=np.append([1000.0],abs(self.__monoisotIons.mz[1:len(self.__monoisotIons)].to_numpy()-self.__monoisotIons.mz[0:-1].to_numpy())/self.__monoisotIons.mz[1:len(self.__monoisotIons)].to_numpy()*1E6)
        self.__monoisotIons['massGroup']=(self.__monoisotIons.dmz_ppm>dmzThresClustering).apply(int).cumsum()            
        self.__monoisotIons['massGroupPoints']=self.__monoisotIons.groupby(['massGroup']).num.transform('count')

        self.__monoisotIons=self.__monoisotIons[self.__monoisotIons.massGroupPoints>=self.__asAssesmentMinNPoints].sort_values(['massGroup','tR']).reset_index(drop=True)
        self.__monoisotIons['massGroup']=list(chain.from_iterable([[idx+1]*mzgrp[1].massGroupPoints for idx,mzgrp in enumerate(self.__monoisotIons[['massGroup','massGroupPoints']].drop_duplicates().iterrows())]))
        self.__monoisotIons.to_csv(self.__monoIsotopicAssesmentsFile,sep="\t",index=False)

        ## Clustering by tR
        __mainIons=self.__monoisotIons[self.__monoisotIons.polarity==polarity][['mz','massGroup']].groupby(['massGroup'],as_index=False).agg({'mz':['mean','std']})['mz']
        __idxIntMax=self.__monoisotIons[self.__monoisotIons.polarity==polarity][['intensity','massGroup']].groupby(['massGroup'],as_index=False).idxmax()['intensity'].to_list()
        __mainIons['tRmax']=self.__monoisotIons.tR[__idxIntMax].to_list()
        __mainIons['Intmax']=self.__monoisotIons.intensity[__idxIntMax].to_list()
        __mainIons['patternCount2max']=self.__monoisotIons.patternCount[__idxIntMax].to_list()
        __mainIons['numOfIsoSignals']=self.__monoisotIons.numOfIsoSignals[__idxIntMax].to_list()
        __mainIons['massGroup']=range(1,len(__mainIons)+1)
        __mainIons=__mainIons.sort_values(['tRmax']).reset_index(drop=True)
        __mainIons['dtR']=np.append([10.0],abs(__mainIons.tRmax[1:len(__mainIons)].to_numpy()-__mainIons.tRmax[0:-1].to_numpy()))
        __mainIons['tRGroup']=(__mainIons.dtR>tRThresClustering).apply(int).cumsum()
        __mainIons=__mainIons.sort_values(['tRGroup','mean']).reset_index(drop=True)
        __mainIons.to_csv(self.__allIonsAssessmentsFile,sep="\t",index=False) 


        ## Clustering ion products: elimination of ionized species produced at the ionization source
        __commonsIF=self.__candidateMoleculesManager.commonIonSpecies.copy()
        __commonsIF=__commonsIF[(__commonsIF.polarity==polarity)].dropna().sort_values(['polfreq'],ascending=False).reset_index(drop=True)
        __commonsIF=__commonsIF[__commonsIF.polfreq>ionFFreq].reset_index(drop=True)


        IF_array=np.array([np.ones(len(__commonsIF))*__commonsIF.z_apported.to_numpy(),-1.0*__commonsIF.mz_apported.to_numpy()])
        MI_array=np.array([__mainIons['mean'].to_numpy(),np.ones(len(__mainIons))]).transpose()
        __completeCandidateIons=pd.DataFrame(np.matmul(MI_array,IF_array),columns=__commonsIF.ionSpecies.to_list())
        __completeCandidateIons['tRGroup']=__mainIons.tRGroup
        __completeCandidateIons['massGroup']=__mainIons.massGroup
        __completeCandidateIons['mz']=__mainIons['mean']
        __completeCandidateIons['mzIdx']=__mainIons.index.to_list()
        __completeCandidateIons['numOfIsoSignals']=__mainIons['numOfIsoSignals']

        tmpDF=__completeCandidateIons.copy().set_index('massGroup').iloc[:,0:len(__commonsIF)].stack().reset_index()
        tmpDF=tmpDF.set_index('massGroup').join(__completeCandidateIons[['massGroup','tRGroup','mz','mzIdx','numOfIsoSignals']].set_index('massGroup')).reset_index().rename(columns={'level_1':'ionSpecies',0:'baseMZ'}).sort_values(['tRGroup','baseMZ']).reset_index(drop=True)
        tmpDF['d_basemz']=np.append([1000],abs(tmpDF.baseMZ[1:].to_numpy()-tmpDF.baseMZ[0:-1].to_numpy())/tmpDF.baseMZ[0:-1].to_numpy()*1E6)
        tmpDF['basemzGroup']=(tmpDF.d_basemz>dmzThresClustering).apply(int).cumsum()
        tmpDF=tmpDF.set_index('basemzGroup').join(tmpDF[['mz','basemzGroup']].groupby('basemzGroup').count().rename(columns={'mz':'ionSpeciesCount'})).reset_index()

        ionToDrop=[]
        candidateIonSpeciesInfo=pd.DataFrame()
        candidateIonSpeciesInfo2=pd.DataFrame()

        __protonicForm="+H"
        if self.polarity=="-": __protonicForm="-H"
        
        for ionFomerGrp in tmpDF.ionSpeciesCount.sort_values(ascending=False).drop_duplicates().to_list():

            if ionFomerGrp!=1:
                for iifg in tmpDF[(tmpDF.ionSpeciesCount==ionFomerGrp)].groupby(['basemzGroup']):              
                    if any(iifg[1].ionSpecies==__protonicForm) :
                        ifl=iifg[1].ionSpecies.to_list()
                        candidateIonSpeciesInfo=pd.concat([candidateIonSpeciesInfo,
                                                          pd.DataFrame({"mzIdx":[iifg[1][iifg[1].ionSpecies==__protonicForm].mzIdx.iloc[0]],
                                                                       "ionSpecies":[";".join(ifl)]
                                                                       })
                                                          ],sort=False,ignore_index=True)
            else:
                for iifg in tmpDF[(tmpDF.ionSpeciesCount==1)].groupby(['basemzGroup']):              
                    if any(iifg[1].ionSpecies==__protonicForm):
                        if iifg[1].numOfIsoSignals.iloc[0]>=3:
                            ifl=iifg[1].ionSpecies.to_list()
                            ifl[0]=";"+ifl[0]
                            ifl[-1]=ifl[-1]+";"
                            candidateIonSpeciesInfo2=pd.concat([candidateIonSpeciesInfo2,
                                                               pd.DataFrame({"mzIdx":[iifg[1][iifg[1].ionSpecies==__protonicForm].mzIdx.iloc[0]],
                                                                             "ionSpecies":[";".join(ifl)]
                                                                             })
                                                            ],sort=False,ignore_index=True)

                            
        candidateIonSpeciesInfo=pd.concat([candidateIonSpeciesInfo,candidateIonSpeciesInfo2],sort=False,ignore_index=True)       
        tmpDF=__mainIons.join(candidateIonSpeciesInfo,how="inner")
        tmpDF.to_csv(self.__completeChromChemSpaceFile,sep="\t",index=False)
        self.__filteredCandidateIons=__completeCandidateIons.copy()
        self.__filteredCandidateIons=self.__filteredCandidateIons.set_index('mzIdx').join(candidateIonSpeciesInfo.set_index('mzIdx'))
        self.__filteredCandidateIons=self.__filteredCandidateIons.iloc[tmpDF.mzIdx.to_list()].sort_values('tRGroup',ascending=False).reset_index(drop=False)
        self.__filteredCandidateIons=self.__filteredCandidateIons.set_index('massGroup',drop=False).join(__mainIons[['massGroup','Intmax','patternCount2max']].set_index('massGroup')).reset_index(drop=True)
        self.__filteredCandidateIons.to_csv(self.__filteredIonsAssessmentFile,sep="\t",index=False)      
        __idx=self.__filteredCandidateIons.patternCount2max.sort_values().to_list()
        self.allCandidateAnalyticalSignals=self.allCandidateAnalyticalSignals.copy().set_index('patternCount',drop=False).loc[__idx].reset_index(drop=True)
        self.__monoisotIons=pd.DataFrame()


        return 

                
    def searchSingleNontargetedCompound(self,ionNum,assessFragmentIons=False):
        self.__allCurrentProbableFragmentIons=pd.DataFrame()
        self.__currentCandidateStructuralFormulas=pd.DataFrame()

        patternCount=self.__filteredCandidateIons[self.__filteredCandidateIons.massGroup==ionNum].patternCount2max.iloc[0]
        __expIsotopicPattern=self.allCandidateAnalyticalSignals.copy()
        __expIsotopicPattern=__expIsotopicPattern[__expIsotopicPattern.patternCount==patternCount]        
        ionSpeciesInfo=self.__filteredCandidateIons[self.__filteredCandidateIons.massGroup==ionNum].copy().drop(columns=['mzIdx','tRGroup','massGroup','Intmax','patternCount2max'])
        self.currentIsotopicPatternIdx=__expIsotopicPattern.patternSectionIdx.iloc[0]
        self.setCurrentCandidateIsoPattern(__expIsotopicPattern.reset_index(drop=True),'+H',ionSpeciesInfo.iloc[0,0]) 
        self.__mzBasedSearch(ionNum)

        if self.__currentCandidateStructuralFormulas.empty: return

        self.__detected+=1
        self.__candidateStructureInferences()
        __filteringFeatures=self.__currentCandidateStructuralFormulas[['p_acceptableSignal',
                                                                       'p_vectorialProj',
                                                                       'p_tR_inference',
                                                                       'p_amenability']].copy().apply(max)

        qltyState=self.__currentCandidateStructuralFormulas.qltySignal.iloc[0]
        if ((qltyState==True) & \
            (__filteringFeatures['p_vectorialProj']>=self.vectorialProjProbThreshold) & \
            (__filteringFeatures['p_tR_inference']>=self.tRInferenceProbThreshold) ):
        
            self.searchNontargetedProbableFragmentIons()
            if assessFragmentIons: self.__fragmentsIonsAssessment()
            self.__saveDectectionInfo(ionNum)
        else:
            self.__currentCandidateStructuralFormulas['p_ms2']=np.nan
            self.__currentCandidateStructuralFormulas['numMS2Fragments']=0
            self.__currentCandidateStructuralFormulas['p_ms2Qlty']=np.nan


            
        self.__allDetectedCandidateStructuralFormulas=pd.concat([self.__allDetectedCandidateStructuralFormulas,self.__currentCandidateStructuralFormulas],sort=False,ignore_index=True)
        self.__allDetectedCandidateStructuralFormulas.to_csv(self.__ionsDetectionFile,sep="\t",index=False)

        return

    def searchNontargetedCompounds(self,assessFragmentIons=True,numOfIsoSignals=2,dmzThresClustering=2.0,patternRange=[0,0],forceSearch=False):

        if self.__filteredCandidateIons.empty: return

        if not(os.path.exists(self.resultsPath)):
            self.resultsPath=self.resultsPath

        if not(forceSearch):
            if os.path.exists(self.__ionsDetectionFile): return

        if os.path.exists(self.__nontargetedXICsFile): os.remove(self.__nontargetedXICsFile)
        if os.path.exists(self.__ionsDetectionFile): os.remove(self.__ionsDetectionFile)
        if os.path.exists(self.__nontargetedIsotopicPatternsFile): os.remove(self.__nontargetedIsotopicPatternsFile)
        if os.path.exists(self.__nontargetedXICsFile): os.remove(self.__nontargetedXICsFile)
        if os.path.exists(self.__fxicsFile): os.remove(self.__fxicsFile)
        if os.path.exists(self.__fragmentIonsFile): os.remove(self.__fragmentIonsFile)
        if os.path.exists(self.__fragmentIonsAnnotationsFile): os.remove(self.__fragmentIonsAnnotationsFile)
        if os.path.exists(self.__enlapsedTimeFile): os.remove(self.__enlapsedTimeFile)

        self.__numOfdetectedCompounds=0
        self.numOfIsoSignals=numOfIsoSignals
        self.isoSignalAssessment_mzAcc=dmzThresClustering
        self.__allDetectedCandidateStructuralFormulas=pd.DataFrame()
        self.__identProcessingtime=pd.DataFrame()
        gstartTime = time.monotonic()
        __compsIdxList=self.__filteredCandidateIons.sort_values('Intmax',ascending=False).massGroup.to_list()
        if patternRange[1]<=0: patternRange[1]=len(__compsIdxList)
        __compsIdxList=__compsIdxList[patternRange[0]:patternRange[1]]
        pbarSpects = tqdm(total=len(__compsIdxList), bar_format='{l_bar}{bar:100}{r_bar}{bar:-5b}')
        count=0

        
        for ionNum in __compsIdxList:
            count+=1
            startTime = time.monotonic()
            clear_output(wait=False)
            os.system('clear')
            pbarSpects.set_description(f"Non-targeted processing (njobs: {self.njobs}): {count}/{len(__compsIdxList)} - {os.path.basename(self.rawFile)}  - Detected: {self.__detected}")      
            self.searchSingleNontargetedCompound(ionNum,assessFragmentIons)

            enlapsed_time = time.monotonic() - startTime
            self.__identProcessingtime=pd.concat([self.__identProcessingtime,
                                                    pd.DataFrame({"ionNum":[ionNum],
                                                                  "processTime":[enlapsed_time]
                                                                  })],sort=False,ignore_index=True)
            self.__identProcessingtime.to_csv(self.__enlapsedTimeFile,sep="\t",index=False)
            pbarSpects.update(1)
                
        clear_output(wait=True) 
        pbarSpects.bar_format
        pbarSpects.close()
            
        enlapsed_time = time.monotonic() - gstartTime
        print(f"Global non-targeted search time: {round(enlapsed_time,3)} s")
        self.__identProcessingtime['totalTime']=enlapsed_time
        self.__identProcessingtime.to_csv(self.__enlapsedTimeFile,sep="\t",index=False)
        self.__allDetectedCandidateStructuralFormulas.to_csv(self.__ionsDetectionFile,sep="\t",index=False)

