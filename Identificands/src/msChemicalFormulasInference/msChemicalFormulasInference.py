# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import os
import subprocess
import io
import time


import pandas as pd
#pd.options.mode.copy_on_write = True
import numpy as np
import intervals
from scipy import special
import matplotlib.pyplot as plt
from collections import Counter
import sqlite3


sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from msIsotopicCluster import msIsotopicCluster
from msMolecule import msMolecule

class msChemicalFormulasInference():

    def __init__(self):
        self.isoPatternsGenerator=msIsotopicCluster()
        self.molFormulas=msMolecule()
        
        #Formula generator and search parameters
        self.__mzAccMolFormulaGenerator=5.0
        self.__mzAccDBSearch_ppm=5.0
        self.__isotopicSignals_mzErr_ppm=[5.0,5.0] 
        self.__isotopicSignals_intensityErr=[0.01,0.35,0.35] 

        #Screening parameters
        self.__chromPeakProbThreshold=0.3
        self.__vectorialProjProbThreshold=0.3
        self.__tRInferenceProbThreshold=0.2
        self.__tRProbThreshold=0.001 
        self.__amenabilityInferenceProbThreshold=0.42
        self.__dtR=0.1
        self.__isCompListUsed=False
        
        #Current cadidate attributes
        self.__isoPatternToAssess=pd.DataFrame()
        self.__currentIsoPatternCharge=1.0
        self.__numberOfIsotopicalSignals=6
        self.__candidateInteractionProductFormulas=pd.DataFrame()
        self.__patternsIntensityMtx=np.empty((0,self.__numberOfIsotopicalSignals))
        self.__patternsMZMtx=np.empty((0,self.__numberOfIsotopicalSignals))
        self.__patternsVecProjectionErrors=pd.DataFrame()
        self.__patternsSVDErrors=pd.DataFrame()
        self.__patternsIntensitySimilarity=pd.DataFrame()
        self.__candidateStructuralFormulas=pd.DataFrame()

        #Chemical Space DB
        self.__CRLPath=os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],'msData/')
        self.__CRL="chemicalSpace_CRL.db"


    @property
    def candidateInteractionProductFormulas(self):
        return self.__candidateInteractionProductFormulas


    @property
    def mzAccMolFormulaGenerator(self):
        return self.__mzAccMolFormulaGenerator

    @mzAccMolFormulaGenerator.setter
    def mzAccMolFormulaGenerator(self,value):
        self.__mzAccMolFormulaGenerator=value

    @property
    def mzAccDBSearch(self):
        return self.__mzAccDBSearch_ppm

    @mzAccDBSearch.setter
    def mzAccDBSearch(self,value):
        self.__mzAccDBSearch_ppm=value

    @property
    def patternsVecProjectionErrors(self):
        return self.__patternsVecProjectionErrors

    @property
    def patternsOrthogonalProjectionErrors(self):
        return self.__patternsSVDErrors

    @property
    def patternsIntensitySimilarity(self):
        return self.__patternsIntensitySimilarity

    @property
    def currentCandidateFormulas(self):
        return self.__candidateFormulas
    

    @property
    def CRLPath(self):
        return self.__CRLPath

    @CRLPath.setter
    def CRLPath(self,value):
        self.__CRLPath=value

    @property
    def CRL(self):
        return self.__CRL

    @CRL.setter
    def CRL(self,value):
        self.__CRL=value

    @property
    def  candidateStructuralFormulas(self):
        return self.__candidateStructuralFormulas
              

    @property
    def isoPatternToAssess(self):
        return self.__isoPatternToAssess

    @isoPatternToAssess.setter
    def isoPatternToAssess(self,value):
        self.__isoPatternToAssess=value

    @property
    def currentIsoPatternCharge(self):
        return self.__currentIsoPatternCharge

    @property
    def patternsIntensityMtx(self):
        return self.__patternsIntensityMtx

    @patternsIntensityMtx.setter
    def patternsIntensityMtx(self,value):
        self.__patternsIntensityMtx=value

    @property
    def patternsMZMtx(self):
        return self.__patternsMZMtx

    @patternsMZMtx.setter
    def patternsMZMtx(self,value):
        self.__patternsMZMtx=value
    
    @property
    def chromPeakProbThreshold(self):
        return self.__chromPeakProbThreshold

    @chromPeakProbThreshold.setter
    def chromPeakProbThreshold(self,value):
        self.__chromPeakProbThreshold=value

    @property
    def vectorialProjProbThreshold(self):
        return self.__vectorialProjProbThreshold

    @vectorialProjProbThreshold.setter
    def vectorialProjProbThreshold(self,value):
        self.__vectorialProjProbThreshold=value

    @property
    def tRInferenceProbThreshold(self):
        return self.__tRInferenceProbThreshold

    @tRInferenceProbThreshold.setter
    def tRInferenceProbThreshold(self,value):
        self.__tRInferenceProbThreshold=value

    @property
    def tRProbThreshold(self):
        return self.__tRProbThreshold

    @tRProbThreshold.setter
    def tRProbThreshold(self,value):
        self.__tRProbThreshold=value
        
    @property
    def amenabilityInferenceProbThreshold(self):
        return self.__amenabilityInferenceProbThreshold

    @amenabilityInferenceProbThreshold.setter
    def amenabilityInferenceProbThreshold(self,value):
        self.__amenabilityInferenceProbThreshold=value

    
    @property
    def dtR(self):
        return self.__dtR

    @dtR.setter
    def dtR(self,value):
        self.__dtR=value

    @property
    def isotopicSignals_mzErr(self):
        return self.__isotopicSignals_mzErr_ppm

    @isotopicSignals_mzErr.setter
    def isotopicSignals_mzErr(self,value):
        self.__isotopicSignals_mzErr_ppm=value

    @property
    def isotopicSignals_intensityErr(self):
        return self.__isotopicSignals_intensityErr

    @isotopicSignals_intensityErr.setter
    def isotopicSignals_intensityErr(self,value):
        self.__isotopicSignals_intensityErr=value
        
    @property
    def isCompListUsed(self):
        return self.__isCompListUsed

    @isCompListUsed.setter
    def isCompListUsed(self,value):
        self.__isCompListUsed=value       

    def getIsoPatternsForCandidateMolFormulas(self):
        self.__patternsIntensityMtx=np.empty((0,self.__numberOfIsotopicalSignals))
        self.__patternsMZMtx=np.empty((0,self.__numberOfIsotopicalSignals))


        if self.__candidateInteractionProductFormulas.empty:  return

        
        self.__candidateFormulas=self.__candidateInteractionProductFormulas.copy()[['Formula','charge','group']].drop_duplicates().sort_values(['group']).reset_index(drop=True)
        for i,formula in enumerate(self.__candidateFormulas.Formula):

            self.isoPatternsGenerator.clusterCharge=self.__candidateFormulas.iloc[0].charge           
            self.isoPatternsGenerator.atomicComposition=self.molFormulas.getFlattenMolecularFormulaDF(formula)
            ipatt=self.isoPatternsGenerator.theoreticalIsoPattern           
            nlack=self.isoPatternsGenerator.isoPatternPeaksNumCalculator-len(ipatt)


            if nlack>0: 
                ipatt=pd.concat([ipatt,pd.DataFrame({'mz':ipatt.mz.iloc[-1]+range(1,nlack+1),'intensity':[0.01]*nlack,'z':[ipatt.z.iloc[0]]*nlack})],sort=False,ignore_index=True)
                

            self.__patternsIntensityMtx= np.append(self.__patternsIntensityMtx,[ipatt.intensity.to_numpy()], axis=0)
            self.__patternsMZMtx= np.append(self.__patternsMZMtx,[ipatt.mz.to_numpy()], axis=0)

        self.__candidateFormulas['interactionProductExactMass']=self.__patternsMZMtx[:,0]        
        return 


    def p_isoPattError(self,m_z):
        denF=1.020426913849308 # np.sqrt(2)*s_z
        int_interval=intervals.closed(m_z-np.sqrt(2),m_z+np.sqrt(2)) & intervals.closed(-np.sqrt(2),np.sqrt(2))
        p=0.0
        if not(int_interval.is_empty()):
            p=0.5*(special.erf((m_z-int_interval.lower)/denF)-special.erf((m_z-int_interval.upper)/denF))/0.95

        return p

    
        
    def assessIsoPatternByIsoPatternsProjection(self,numOfIsoSignals=3,evalCandidateFormulas=True):

        if not(self.__patternsIntensityMtx.any()): return

        isoPatternToAssess__=self.__isoPatternToAssess[0:min(len(self.__isoPatternToAssess),numOfIsoSignals)].copy()
        isoSignalsToSelect=(isoPatternToAssess__.mz-isoPatternToAssess__.mz[0]).round().astype(int).to_list()
        __nTIS=self.__patternsMZMtx.shape[1]

        ## MZ scale
        mz_experimental=isoPatternToAssess__.mz.to_numpy()
        mz_refs=self.__patternsMZMtx[:,isoSignalsToSelect]
        d=1/np.einsum("ij,ij->i",mz_refs,mz_refs)
        mz_projector= np.matmul(mz_refs.T,np.diag(d)).T       
        mz_errors=np.concatenate([[self.__isotopicSignals_mzErr_ppm[0]],[self.__isotopicSignals_mzErr_ppm[1]]*(__nTIS-1)]).reshape(1,__nTIS)/1E6       
        mz_errors=np.array([mz_errors[0,isoSignalsToSelect]])
        mz_errors=np.matmul(np.ones((len(mz_refs),1)),mz_errors) 
        dmz_err=np.multiply(mz_refs,mz_errors)
        D_mz_proj=np.einsum("ij,ij->i",dmz_err,mz_projector)
        mzProjections=np.matmul(mz_experimental,mz_projector.T)
        Err_mz=abs((mzProjections-1)/D_mz_proj)


        ## Intensity scale
        int_experimental=isoPatternToAssess__.intensity.to_numpy()
        int_experimental=int_experimental/int_experimental.max()
        int_refs=self.__patternsIntensityMtx[:,isoSignalsToSelect]
        d=1/np.einsum("ij,ij->i",int_refs,int_refs)
        int_projector= np.matmul(int_refs.T,np.diag(d)).T
        int_errors=np.matrix(self.__isotopicSignals_intensityErr+[self.__isotopicSignals_intensityErr[-1]]*(__nTIS-len(self.__isotopicSignals_intensityErr)))
        int_errors=np.array(int_errors[0,isoSignalsToSelect])
        int_errors=np.matmul(np.ones((len(int_refs),1)),int_errors) 
        dint_err=np.multiply(int_refs,int_errors)
        D_int_proj=np.einsum("ij,ij->i",dint_err,int_projector)
        intProjections=np.matmul(int_experimental,int_projector.T)
        Err_int=abs((intProjections-1)/D_int_proj)
        
        __combError=(Err_mz**2+Err_int**2)**0.5
        __p=[self.p_isoPattError(cErr) for cErr in __combError]
        

        if evalCandidateFormulas:
            if len(self.__candidateInteractionProductFormulas)!=len(Err_int):
                __gkeying=pd.DataFrame({"intErr":Err_int,
                                        "mzErr":Err_mz,
                                        "cErr":__combError,
                                        "p":__p})
                __gkeying=__gkeying.join(self.__candidateInteractionProductFormulas.set_index('group')).reset_index(drop=True)
                self.__patternsVecProjectionErrors=pd.DataFrame({'interactionProductFormula':__gkeying.Formula,
                                                      'vectorialProj_intensityError':__gkeying.intErr,
                                                      'vectorialProj_mzError':__gkeying.mzErr,
                                                      'vectorialProj_combinedError':__gkeying.cErr,
                                                      'p_vectorialProj':__gkeying.p,
                                                      'dExactMass_ppm':__gkeying.dExactMass_ppm
                                                       })
            else:
                self.__patternsVecProjectionErrors=pd.DataFrame({'interactionProductFormula':self.__candidateInteractionProductFormulas.Formula,
                                                  'vectorialProj_intensityError':Err_int,
                                                  'vectorialProj_mzError':Err_mz,
                                                  'vectorialProj_combinedError':__combError,
                                                  'p_vectorialProj':__p,
                                                  'dExactMass_ppm':self.__candidateInteractionProductFormulas.dExactMass_ppm
                                                  })
        else:
            self.__patternsVecProjectionErrors=pd.DataFrame({'vectorialProj_intensityError':Err_int,
                                                                 'vectorialProj_mzError':Err_mz,
                                                                 'vectorialProj_combinedError':__combError,
                                                                 'p_vectorialProj':__p
                                                                 })

                
        self.__patternsVecProjectionErrors["rankSource"]=self.__patternsVecProjectionErrors.index
        self.__patternsVecProjectionErrors=self.__patternsVecProjectionErrors.sort_values(['vectorialProj_combinedError']).reset_index(drop=True)

    def assessIsoPatternIntensitySimilarity(self,numOfIsoSignals=3,thres=1.75):

        if not(self.__patternsIntensityMtx.any()):
            return

        isoPatternToAssess__=self.__isoPatternToAssess[0:min(len(self.__isoPatternToAssess),numOfIsoSignals+1)].copy()

        referenceIntensity=np.array(isoPatternToAssess__.intensity.to_list()+[0.0]*(6-len(isoPatternToAssess__.intensity)))
        referenceIntensity=referenceIntensity/referenceIntensity.max()

        errorIntensityMtx=self.__patternsIntensityMtx-referenceIntensity        
        intensitySimilarity=np.log2(1.0/np.sum(errorIntensityMtx**2,axis=1)/len(referenceIntensity))
        
        __p=1.0/(1.0+np.exp(-(intensitySimilarity-thres)))        

        if len(self.__candidateInteractionProductFormulas)!=len(intensitySimilarity):
            __gkeying=pd.DataFrame({"intSim":intensitySimilarity,
                                    "p":__p})
            __gkeying=__gkeying.join(self.__candidateInteractionProductFormulas.set_index('group')).reset_index(drop=True)

            self.__patternsIntensitySimilarity=pd.DataFrame({'interactionProductFormula':__gkeying.Formula,
                                                  'intensitySimilarity':__gkeying.intSim,
                                                  'p_intensitySimilarity':__gkeying.p,
                                                  'dExactMass_ppm':__gkeying.dExactMass_ppm
                                                   })
        else:
            self.__patternsIntensitySimilarity=pd.DataFrame({'interactionProductFormula':self.__candidateInteractionProductFormulas.Formula,
                                                  'intensitySimilarity':intensitySimilarity,
                                                  'p_intensitySimilarity':__p,
                                                  'dExactMass_ppm':self.__candidateInteractionProductFormulas.dExactMass_ppm
                                                   })
            
        self.__patternsIntensitySimilarity["rankSource"]=self.__patternsIntensitySimilarity.index
        self.__patternsIntensitySimilarity=self.__patternsIntensitySimilarity.sort_values(['intensitySimilarity'],ascending=False).reset_index(drop=True)        
       
        
    def plotPatternsVecProjectionErrors(self,numCandidates=None):
        
        if self.__patternsVecProjectionErrors.empty:
            return

        __patternsVecProjectionErrors=self.__patternsVecProjectionErrors.copy()
        if numCandidates!=None:
            __patternsVecProjectionErrors=self.__patternsVecProjectionErrors[0:numCandidates].copy()
            
        plt.rcdefaults()
        plt.scatter(range(len(__patternsVecProjectionErrors.vectorialProj_combinedError)),
                    __patternsVecProjectionErrors.vectorialProj_combinedError,
                    marker="o",
                    color="skyblue",
                    s=150)
        plt.xlabel("Rank")
        plt.ylabel("Projection based combined error")
        plt.title("Projection based error")


        for i,txt in enumerate(__patternsVecProjectionErrors.rankSource):
            plt.annotate(txt, (i,__patternsVecProjectionErrors.vectorialProj_combinedError[i]))


        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.rcdefaults()

        
    def plotIsoPatternIntensitySimilarity(self,numCandidates=None):
        
        if self.__patternsIntensitySimilarity.empty:
            return

        __patternsIntensitySimilarity=self.__patternsIntensitySimilarity.copy()
        if numCandidates!=None:
            __patternsIntensitySimilarity=self.__patternsIntensitySimilarity[0:numCandidates].copy()


            
        plt.rcdefaults()
        plt.scatter(range(len(__patternsIntensitySimilarity.intensitySimilarity)),
                    __patternsIntensitySimilarity.intensitySimilarity,
                    marker="o",
                    color="skyblue",
                    s=150)
        plt.xlabel("Rank")
        plt.ylabel("Isotopic pattern intensity similarity")
        plt.title("Intensity similarity")

        for i,txt in enumerate(__patternsIntensitySimilarity.rankSource):
            plt.annotate(txt, (i,__patternsIntensitySimilarity.intensitySimilarity[i]))


        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.rcdefaults()

        

    def plotCandidateFormulaIsoPattern(self,candidateFormulaRank=0,metrics=None):

        if metrics=="orthogonal":
            mtrcs="Orthogonal dist."
            if self.__patternsSVDErrors.empty: return
            rankedFormulas=self.__patternsSVDErrors.copy()
            
        elif metrics=="similarity":
            mtrcs="Int. similarity"
            if self.__patternsIntensitySimilarity.empty: return
            rankedFormulas=self.__patternsIntensitySimilarity.copy()
        elif metrics=="vectorial":
            mtrcs="Vectorial dist."
            if self.__patternsVecProjectionErrors.empty: return
            rankedFormulas=self.__patternsVecProjectionErrors.copy()
        else:
            mtrcs="Rank from source"

            
        if mtrcs=="Rank from source":
            Ho_idx=candidateFormulaRank%len(self.__candidateInteractionProductFormulas)
            iPF=self.__candidateInteractionProductFormulas.iloc[Ho_idx].Formula
            dEM=np.round(self.__candidateInteractionProductFormulas.iloc[Ho_idx].dExactMass_ppm,2)
            
        else:
            Ho_isotopicPattenInfo=rankedFormulas.iloc[candidateFormulaRank%len(rankedFormulas)]  
            Ho_idx=self.__candidateInteractionProductFormulas['group'].iloc[Ho_isotopicPattenInfo.rankSource]           
            iPF=Ho_isotopicPattenInfo.interactionProductFormula
            dEM=np.round(Ho_isotopicPattenInfo.dExactMass_ppm,2)

            
        Ho_isotopicPatten=pd.DataFrame({'mz':self.__patternsMZMtx[Ho_idx,:],
                                'intensity':self.__patternsIntensityMtx[Ho_idx,:]})

        plt.rcdefaults()
        plt.plot(self.__isoPatternToAssess.mz,self.__isoPatternToAssess.intensity/self.__isoPatternToAssess.intensity.max(),'o-',color='blue',label="Experimental isotopic pattern")
        plt.plot(Ho_isotopicPatten.mz,Ho_isotopicPatten.intensity,'o-',color='orange',markersize=5,label=f"Theoretical pattern {Ho_idx} for {iPF}")
        plt.title(f"Experimental vs. candidate isotopic pattern:\n m/z error={dEM} ppm\n Projection rank: {candidateFormulaRank%len(self.__candidateInteractionProductFormulas)} (m:{mtrcs})")
        plt.xlabel("m/z (Th)")
        plt.ylabel("Abundance")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.rcdefaults()
        

    def plotIsotopicPatternVecAssesment(self,isoPatternIdx=0):

        if self.__patternsVecProjectionErrors.empty: return
        rankedFormulas=self.__patternsVecProjectionErrors.copy()

        Ho_idx=isoPatternIdx%len(rankedFormulas)
        Ho_isotopicPattenInfo=rankedFormulas.iloc[Ho_idx]  
        Ho_isotopicPatten=pd.DataFrame({'mz':self.__patternsMZMtx[Ho_idx,:],
                                'intensity':self.__patternsIntensityMtx[Ho_idx,:]})

        plt.rcdefaults()
        plt.plot(self.__isoPatternToAssess.mz,self.__isoPatternToAssess.intensity/self.__isoPatternToAssess.intensity.max(),'o-',color='blue',label="Experimental isotopic pattern")
        plt.plot(Ho_isotopicPatten.mz,Ho_isotopicPatten.intensity,'o-',color='orange',markersize=5,label=f"Theoretical pattern {Ho_idx}")
        plt.title(f"Experimental vs. candidate isotopic pattern:\n  Projection rank: {Ho_idx}")
        plt.xlabel("m/z")
        plt.ylabel("Abundance")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.rcdefaults()

        
    def dbStructuralFormulasSearch(self,molecularFormula):

        self.__candidateStructuralFormulas=pd.DataFrame()
        
        conexion=sqlite3.connect(os.path.join(self.__CRLPath,self.__CRL))
        smilesQuery=conexion.execute("select * from chemicalSpaceReferenceLibrary where molecularFormula=?",
                                         (molecularFormula,))
        self.__candidateStructuralFormulas=pd.DataFrame(smilesQuery.fetchall(),columns=["IDkey","name","canonicalSmiles","inchikey","molecularFormula","bmonoisotopicMass"])
        self.__candidateStructuralFormulas=self.__candidateStructuralFormulas.drop_duplicates(subset=['canonicalSmiles'])
        
        return
        
    def rankIntersections(self, metrics=0,maxSize=None,asList=True):

        if metrics==0:
            if self.__patternsIntensitySimilarity.empty | self.__patternsVecProjectionErrors.empty: return
            metricsA=self.__patternsIntensitySimilarity.copy()
            metricsB=self.__patternsVecProjectionErrors.copy()
        elif metrics==1:
            if self.__patternsIntensitySimilarity.empty | self.__patternsSVDErrors.empty: return
            metricsA=self.__patternsIntensitySimilarity.copy()
            metricsB=self.__patternsSVDErrors.copy()
        elif metrics==2:
            if self.__patternsSVDErrors.empty | self.__patternsVecProjectionErrors.empty: return
            metricsA=self.__patternsSVDErrors.copy()
            metricsB=self.__patternsVecProjectionErrors.copy()
        else:
            return

        __maxSize=maxSize
        if isinstance(maxSize,type(None)):__maxSize=len(metricsA)
            
        __rankClusters={}
        for idx in range(len(metricsA)):
            set1=set(metricsA.rankSource[0:idx].to_list())
            set2=set(metricsB.rankSource[0:idx].to_list())
            intersections=set1.intersection(set2)
            if len(intersections)>0:
                __rankClusters[str(intersections)]=intersections
                if len(__rankClusters[str(intersections)])>=__maxSize:
                    break
        __rankClusters=dict(zip(range(len(__rankClusters.keys())), list(__rankClusters.values())))
        if asList:
            __rankClusters=list(__rankClusters[0])+[ list(__rankClusters[i+1]-__rankClusters[i])[0]  for i in range(0,len(__rankClusters)-1) if len(__rankClusters[i+1]-__rankClusters[i])==1]
            
        return __rankClusters

    def formulaDBsearch(self,mzTarget=None,molecularFormula=None):

        if not(isinstance(mzTarget,type(None))):
            __searchMass=mzTarget
            __dMZTarget=__searchMass*self.mzAccDBSearch/1E6
            conexion=sqlite3.connect(os.path.join(self.__CRLPath,self.__CRL))
            smilesQuery=conexion.execute(f"select * from chemicalSpaceReferenceLibrary where exactMass between {__searchMass-__dMZTarget} and {__searchMass+__dMZTarget}")

            self.__candidateStructuralFormulas=pd.DataFrame(smilesQuery.fetchall(),columns=["IDkey","name","canonicalSmiles","inchikey","molecularFormula","exactMass"])
            return self.__candidateStructuralFormulas
        elif isinstance(molecularFormula,str):
            conexion=sqlite3.connect(os.path.join(self.__CRLPath,self.__CRL))
            smilesQuery=conexion.execute(f"select * from chemicalSpaceReferenceLibrary where molecularFormula like '{molecularFormula}'")

            self.__candidateStructuralFormulas=pd.DataFrame(smilesQuery.fetchall(),columns=["IDkey","name","canonicalSmiles","inchikey","molecularFormula","exactMass"])
            return self.__candidateStructuralFormulas

            

        
    
    def getStructuralFormulasFromDBMassSearch(self,analyticalSignal):

        self.__candidateStructuralFormulas=pd.DataFrame()
        self.__candidateInteractionProductFormulas=pd.DataFrame()

        if not(analyticalSignal.currentCandidateIsoPattern.empty):
            analyticalSignal.assessChrom4CurrCandidateIsoPattern()

            if analyticalSignal.currentCandidateChromPeakAssessment.empty: return pd.DataFrame()
            exp_dtR=abs(analyticalSignal.currentCandidateChromPeakAssessment.tRmax[0]-analyticalSignal.currentScan_tR) 

            if ( (analyticalSignal.currentCandidateChromPeakAssessment.qltySignal[0]==True) &
                 (exp_dtR<=self.__dtR)):

                self.__isoPatternToAssess=analyticalSignal.currentCandidateIsoPattern
                self.__currentIsoPatternCharge=analyticalSignal.currentCharge

                mzTarget=float(analyticalSignal.mzTarget)
                __ionSpecies=analyticalSignal.currentIonSpecies.iloc[0].copy()
                __ionSpeciesDmz=mzTarget-analyticalSignal.dbSearchMass
                __searchMass=analyticalSignal.dbSearchMass
                __dMZTarget=__searchMass*self.mzAccDBSearch/1E6
                conexion=sqlite3.connect(os.path.join(self.__CRLPath,self.__CRL))
                smilesQuery=conexion.execute(f"select * from chemicalSpaceReferenceLibrary where exactMass between {__searchMass-__dMZTarget} and {__searchMass+__dMZTarget}")
                self.__candidateStructuralFormulas=pd.DataFrame(smilesQuery.fetchall(),columns=["IDkey","name","canonicalSmiles","inchikey","molecularFormula","bmonoisotopicMass"])
                self.__candidateStructuralFormulas=self.__candidateStructuralFormulas.drop_duplicates(subset=['canonicalSmiles'])
                self.__candidateStructuralFormulas['searchMZ']=__searchMass
                self.__candidateStructuralFormulas['ionizedSpecies']=__ionSpecies.formula
                
                self.__candidateInteractionProductFormulas=pd.DataFrame({'eiFormula':[f+__ionSpecies.formula for f in self.__candidateStructuralFormulas.molecularFormula]})
                self.__candidateInteractionProductFormulas['Formula']=[self.molFormulas.getFlattenMolecularFormula(eiformula) for eiformula in self.__candidateInteractionProductFormulas.eiFormula]               
                self.__candidateInteractionProductFormulas['Score']=None
                self.__candidateInteractionProductFormulas['exactmass']=self.__candidateStructuralFormulas.bmonoisotopicMass.to_numpy()+__ionSpeciesDmz               
                self.__candidateInteractionProductFormulas['charge']=__ionSpecies.z
                self.__candidateInteractionProductFormulas['DBE']=None
                self.__candidateInteractionProductFormulas['dExactMass_ppm']=(mzTarget-self.__candidateInteractionProductFormulas.exactmass)/self.__candidateInteractionProductFormulas.exactmass*1E6
                self.__candidateInteractionProductFormulas['abs']=abs(self.__candidateInteractionProductFormulas['dExactMass_ppm'])

                self.__candidateStructuralFormulas['abs']=self.__candidateInteractionProductFormulas['abs'].to_list()
                self.__candidateStructuralFormulas=self.__candidateStructuralFormulas.sort_values(['abs']).reset_index(drop=True).drop(columns=['abs'])     

                self.__candidateInteractionProductFormulas=self.__candidateInteractionProductFormulas.sort_values(['abs']).reset_index(drop=True).drop(columns=['abs'])     
                self.__candidateInteractionProductFormulas['group']=self.__candidateInteractionProductFormulas.groupby(['Formula']).ngroup().to_list()
                
            return 

    
