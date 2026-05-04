import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import intervals
from scipy import special
import sqlite3

sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from msMolecule import msMolecule
from lcmsMeasurements import lcmsMeasurements
from msMLModelsCommon import msMLModelsCommon
from msChemicalFormulasInference import msChemicalFormulasInference
from  msAnalyticalSignals.msAnalyticalSignals import *
from msInteractionProduct import msInteractionProduct

msMLModelsComm=msMLModelsCommon()
msIntProducts=msInteractionProduct()


class msCompoundExplorer(msMolecule,lcmsMeasurements,msChemicalFormulasInference):
    def __init__(self):
        msMolecule.__init__(self)
        lcmsMeasurements.__init__(self)
        msChemicalFormulasInference.__init__(self)
        self.__rawFile=""
        self.__ionizedSpecies=()
        self.__ionizedSpeciesInfo=pd.DataFrame()
        self.__labeledIsoPatternSection=pd.DataFrame()
        self.__compoundSmiles=""
        self.__ridgeInfo=None
        self.__tRInference=pd.DataFrame()
        self.__mz=None
        self.__elemComp=None
        self.__adqPolarity="+"
        self.__sFormulas=None
        self.tR_uncertainty=0.1

        self.isotopicSignals_mzErr_ppm=[5.0,5.0] # ppm
        self.isotopicSignals_intensityErr=[0.01,0.35,0.35] 
        self.__commonIonSpecies=pd.read_csv(os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],"msData/commonIonizedSpecies.tsv"),sep="\t")



    @property
    def rawFile(self):
        return self.__rawFile

    @rawFile.setter
    def rawFile(self,value):
        self.__rawFile=value
        self.MSfile=self.__rawFile
        self.events=self.__rawFile

        self.__adqPolarity=self.filters.copy()
        self.__adqPolarity=self.__adqPolarity[self.__adqPolarity.MSn=="ms1"].polarity.iloc[0]
        self.polarity=self.__adqPolarity
        self.probableFragmentIonsXICs=pd.DataFrame()
        self.probableFragmentIons=pd.DataFrame()
        self.msFragmenter.msAnnotationData=pd.DataFrame()
        self.__tRInference=pd.DataFrame()

    @property
    def adqPolarity(self):
        return self.__adqPolarity

    @adqPolarity.setter
    def adqPolarity(self,value):
        self.__adqPolarity=value
        
    @property
    def compoundSmiles(self):
        return self.smiles

    @compoundSmiles.setter
    def compoundSmiles(self,value):
        self.smiles=value

        if not(isinstance(self.smiles,type(None))):       
            self.__setPolarity()
            if len(self.__ionizedSpecies)>0:
                self.transformFormula(self.__ionizedSpecies[0],self.__ionizedSpecies[1])
                self.__setPolarity()

            self.mzTarget=self.exactMass.exactMW.iloc[0]
            self.__mz=self.mzTarget
            self.probableFragmentIonsXICs=pd.DataFrame()
            self.probableFragmentIons=pd.DataFrame()
            self.__labeledIsoPatternSection=pd.DataFrame()
            self.msFragmenter.msAnnotationData=pd.DataFrame()

    @property
    def mz(self):
        return self.__mz

    @mz.setter
    def mz(self,value):
        self.__mz=value
        self.mzTarget=value
        self.exactMass=pd.DataFrame({'exactMW':[value],'u_exactMW':[None],'mz':[value]})
   
   
    @property
    def ionizedSpecies(self):
        return self.__ionizedSpecies       

    @property
    def ionicTransform(self):
        return self.__ionizedSpeciesInfo

    @ionicTransform.setter
    def ionicTransform(self,value):

        if value=="":
            self.__ionizedSpeciesInfo=pd.DataFrame()
            self.__ionizedSpecies=()
            self._msMolecule__setClusterCharge(0.0)
            self.polarity=self.__adqPolarity
            if not(isinstance(self.smiles,type(None))): self.compoundSmiles=self.smiles
            return
        
        self.__ionizedSpeciesInfo=self.__commonIonSpecies[self.__commonIonSpecies.ionSpecies==value]
        if self.__ionizedSpeciesInfo.empty: return
        self.__ionizedSpecies=(self.__ionizedSpeciesInfo.ionSpecies.iloc[0],self.__ionizedSpeciesInfo.z_apported.iloc[0])
        
        if not(isinstance(self.smiles,type(None))):
            self.transformFormula(self.__ionizedSpecies[0],self.__ionizedSpecies[1])
            self.mzTarget=self.exactMass.exactMW.iloc[0]
            self.__mz=self.mzTarget

        else:
            if self.formula!="":
                self.transformFormula(self.__ionizedSpecies[0],self.__ionizedSpecies[1])
                self.mzTarget=self.exactMass.exactMW.iloc[0]
                self.__mz=self.mzTarget
            else:
                self._msMolecule__setClusterCharge(self.__ionizedSpeciesInfo.z_apported.iloc[0])

            
        self.__setPolarity()
        self.probableFragmentIonsXICs=pd.DataFrame()
        self.probableFragmentIons=pd.DataFrame()
        self.__labeledIsoPatternSection=pd.DataFrame()
        self.msFragmenter.msAnnotationData=pd.DataFrame()
   
    
    @property
    def labeledIsoPatternSection(self):
        return self.__labeledIsoPatternSection

    @property
    def sFormulas(self):
        return self.__sFormulas
    
    def __setPolarity(self):
        self.polarity="+"
        if self.charge<0:
            self.polarity="-"
       

    def commonIonsBuilders(self,asGroupingArray=False,polarity="",ionFFreq=0.1):

        iBdf=msIntProducts.commonIonSpecies
        if asGroupingArray:
            if polarity!="":
                polarity_=polarity
            else:
                polarity_=self.polarity
            
            iBdf=iBdf[(iBdf.polarity==polarity_)].dropna().sort_values(['polfreq'],ascending=False).reset_index(drop=True)
            iBdf=iBdf[iBdf.polfreq>ionFFreq].reset_index(drop=True)
            iBdf=np.array([np.ones(len(iBdf))*iBdf.z_apported.to_numpy(),-1.0*iBdf.mz_apported.to_numpy()])

        return iBdf

            
    def assessChromatographicSignal(self):
        if (os.path.exists(self.rawFile)) & ( not(self.mzTarget is None) ):
            xicHog=self.getXICArray()
            if isinstance(xicHog,tuple):
                return msMLModelsComm.classifyChromatographicSignal(xicHog[0])
        return pd.DataFrame()
            
    def getMassSpectrum(self,isPlotShown=False,ridgeInfo=None):
    
        if (os.path.exists(self.rawFile)) & ( not(self.mzTarget is None) ):           

            if isinstance(ridgeInfo,type(None)):
                self.__ridgeInfo=self.getXICRidgeInfo()
            else:
                self.__ridgeInfo=ridgeInfo

            
            if isinstance(self.__ridgeInfo,type(None)): return
            
            self.currMS1ScanNum2tRmax=self.eventsDF[self.eventsDF.tR==self.__ridgeInfo.time.iloc[0]].iloc[0].nScan
            
            __massSpectrum=self.getScan(self.currMS1ScanNum2tRmax)

            if isPlotShown:
                plt.rcdefaults()
                plt.clf()
                markerline, stemlines, baseline=plt.stem(__massSpectrum.mz,__massSpectrum.intensity,markerfmt='none',basefmt="none")
                plt.axvline(x=self.mzTarget,
                            ymin=0.0,ymax=1.0,color='gray',
                            linestyle = ':', alpha = 0.5)
                plt.axvline(x=self.mzTarget+7.0,
                            ymin=0.0,ymax=1.0,color='gray',
                            linestyle = ':', alpha = 0.5)

                stemlines.set_alpha(0.5)
                baseline.set_color('gray')
                baseline.set_alpha(0.25)
                baseline.set_linewidth(0.1)
                plt.xlabel('m/z')
                plt.ylabel('Intensity')
                pltName="ms_"+self.fileName
                plt.tight_layout()
                plt.show()
                plt.clf()

            return __massSpectrum

    def getProbableExperimentalIsotopicPattern(self,isPlotShown=False,writeInference=False):
        self.__labeledIsoPatternSection=pd.DataFrame()
        __isoPatternSection=self.getExperimentalIsotopicPatternSection()
        if __isoPatternSection.empty: return pd.DataFrame()     
        if isinstance(__isoPatternSection,type(None)): return pd.DataFrame()
        
        __extractedIsoPattern=evalIsopatterns(__isoPatternSection,self.intensityThreshold,writeInference)
        if not(__extractedIsoPattern.empty):

            __extractedIsoPattern=__extractedIsoPattern[['mz','intensity','rintensity','ion','p_ion']]
            __extractedIsoPattern=__extractedIsoPattern.sort_values("ion").reset_index(drop=True)
            self.__labeledIsoPatternSection=__extractedIsoPattern.set_index('mz').join(__isoPatternSection.set_index('mz'),rsuffix="_",how='outer').reset_index(drop=False)
            self.__labeledIsoPatternSection['ion']=self.__labeledIsoPatternSection['ion'].fillna("S")
            self.__labeledIsoPatternSection['intensity']=self.__labeledIsoPatternSection['intensity_']
            self.__labeledIsoPatternSection['rintensity']=self.__labeledIsoPatternSection['rintensity_']
            self.__labeledIsoPatternSection=self.__labeledIsoPatternSection.drop(columns=['intensity_','rintensity_'])

            __spurious=self.__labeledIsoPatternSection[self.__labeledIsoPatternSection.ion=="S"]

            if isPlotShown:
                plt.rcdefaults()
                plt.clf()
                
                emarkerline, estemlines, ebaseline=plt.stem(__extractedIsoPattern.mz,__extractedIsoPattern.rintensity,
                                                            basefmt="none",linefmt="b-",label="Probable Isotopic Pattern")
                emarkerline.set_markersize(3)
                estemlines.set_alpha(0.5)
                ebaseline.set_color('gray')
                ebaseline.set_linewidth(0.1)

                smarkerline, sstemlines, sbaseline=plt.stem(__spurious.mz,__spurious.rintensity,basefmt="none",linefmt="r-",label="Spurious")
                smarkerline.set_markersize(3)
                sstemlines.set_alpha(0.15)
                sbaseline.set_color('gray')
                sbaseline.set_linewidth(0.1)
                
                
                plt.xlabel('m/z')
                plt.ylabel('Relative Intensity')
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.rcdefaults()
                plt.clf()

        return __extractedIsoPattern
    
            
    def getExperimentalIsotopicPatternSignals(self,isPlotShown=False):

        isoPattern=self.theoreticalIsotopicPattern
        if isoPattern.empty: return
        __isoPatternSection=self.getExperimentalIsotopicPatternSection()
        if isinstance(__isoPatternSection,type(None)): return

        if not(__isoPatternSection.empty):
            __isoPatternSection=__isoPatternSection.iloc[[abs(__isoPatternSection.mz-isoSignal[1].mz).idxmin() for isoSignal in isoPattern.iterrows()]].reset_index(drop=True)
            __isoPatternSection=__isoPatternSection.join(isoPattern,rsuffix="_calculated")
            __isoPatternSection['mzErr_ppm']=[ (iSignal[1].mz-iSignal[1].mz_calculated)/iSignal[1].mz_calculated*1E6 for iSignal in __isoPatternSection.iterrows()]

            __maxInt=__isoPatternSection.intensity[__isoPatternSection.intensity_calculated.idxmax()]
            __isoPatternSection['relIntensityErr']=__maxInt*__isoPatternSection.intensity_calculated
            __isoPatternSection['relIntensityErr']=[ (iSignal[1].intensity-iSignal[1].relIntensityErr)/iSignal[1].relIntensityErr for iSignal in __isoPatternSection.iterrows()]

            __spurious=__isoPatternSection[abs(__isoPatternSection.mzErr_ppm)>self.mzAcc]
            __isoPatternSection=__isoPatternSection[abs(__isoPatternSection.mzErr_ppm)<=self.mzAcc]
            __isoPatternSection['rintensity']=__isoPatternSection.intensity/__isoPatternSection.intensity.max()

            
            if isPlotShown:
                plt.rcdefaults()
                plt.clf()

                #Theorical
                markerline, stemlines, baseline=plt.stem(isoPattern.mz,isoPattern.intensity,basefmt="none",linefmt="g-",label="Theoretical")
                markerline.set_markersize(3)
                markerline.set_color('g')
                stemlines.set_alpha(0.25)
                stemlines.set_color('g')
                baseline.set_linewidth(0.1)

                
                ## Experimental
                emarkerline, estemlines, ebaseline=plt.stem(__isoPatternSection.mz,__isoPatternSection.rintensity,basefmt="none",linefmt="b-",label="Experimental")
                emarkerline.set_markersize(3)
                emarkerline.set_color('b')
                estemlines.set_alpha(0.5)
                ebaseline.set_linewidth(0.1)
                
                plt.xlabel('m/z')
                plt.ylabel('Relative Intensity')
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.rcdefaults()
                plt.clf()

            return __isoPatternSection


    def getCandidateFragmentationProfile(self,mzFragmentsList=None,tR_thres=0.1,ms2_maxtRErr=0.1,p_thres=0.25,ridgeInfo=None):
        
        if (os.path.exists(self.rawFile)) & ( not(self.mzTarget is None) ):

            if isinstance(ridgeInfo,type(None)):
                self.__ridgeInfo=self.getXICRidgeInfo()
            else:
                self.__ridgeInfo=ridgeInfo
            
            if isinstance(self.__ridgeInfo,type(None)):return plotProbableFragmentIonsXIC
            
            self.currMS1ScanNum2tRmax=self.eventsDF[self.eventsDF.tR==self.__ridgeInfo.time.iloc[0]].iloc[0].nScan
            if len(self.getXICArray())==0: return pd.DataFrame()
            self.currentFSXICPeak=self.getXICArray()[1].copy()

            if isinstance(ridgeInfo,type(None)):
                self.tR=self.getXICRidgeInfo().time.iloc[0]
            else:
                self.tR=ridgeInfo.time.iloc[0]
                       
            self.mzTargetFS=self.mzTarget

            __mzFragmentsList=mzFragmentsList
            if isinstance(mzFragmentsList,list):
                __mzFragmentsList=pd.DataFrame({'mz':mzFragmentsList})[['mz']].drop_duplicates().sort_values(['mz'],ascending=False).reset_index(drop=True)
                
            self.searchTargetProbableFragmentIons(self.exactMass.exactMW.iloc[0],self.polarity,[],ms2_maxtRErr,self.mzAcc,__mzFragmentsList,tR_thres=tR_thres,p_thres=p_thres)               
            return self.probableFragmentIons

        
    def annotateProbableFragmentIons(self):
        self.msFragmenter.msAnnotationData=pd.DataFrame()
        self.__currentDetectedFragments=pd.DataFrame()
        self.annotatedFragmentIons=pd.DataFrame()
        if(self.probableFragmentIons.empty): return
        __ms4annotation=self.probableFragmentIons[['mz','intensity']].copy()

        if isinstance(self.smiles,str):
            if len(self.baseFormula)>0:

                self.msFragmenter.msSpectrum4Annotation=__ms4annotation
                __extDB=pd.DataFrame({'MonoisotopicMass':[self.baseExactMass.iloc[0].exactMW],
                                      'Identifier':["001"],
                                      'MolecularFormula':[self.baseFormula],
                                      'SMILES':[self.smiles],
                                      'InChI':[self.inchikey]})
                
                __db_name=f"db_d001_{datetime.datetime.now().strftime('%y%m%d%H%M%S%f')}"
                __extDB.to_csv(__db_name,sep=",",index=False)
                self.msFragmenter.annotateSpectrum(self.smiles,
                                                                self.__ionizedSpecies[0],
                                                                self.baseFormula,
                                                                __db_name,
                                                                __extDB.MonoisotopicMass.iloc[0]
                                                                )
                os.remove(__db_name)
                self.annotatedFragmentIons=self.msFragmenter.msAnnotationData
                return self.msFragmenter.msAnnotationData
        return pd.DataFrame()
        
        

    def getRetentionTimeInference(self,RT_exp=None):
        if not(isinstance(self.__ridgeInfo,type(None))):
            __tRInf=super(msCompoundExplorer,self).getRetentionTimeInference(self.__ridgeInfo.time.iloc[0])
            __tRInf['tR_exp']=round(self.__ridgeInfo.time,3)
            self.__tRInference=__tRInf
        else:
            self.__tRInference=super(msCompoundExplorer,self).getRetentionTimeInference(RT_exp)

        if(self.__tRInference.empty): return self.__tRInference
        
        w_ptR=1.0/(1.0+np.log10(max(self.__tRInference['U_tR']/self.tR_uncertainty)+1E-16))

        self.__tRInference.insert(3,'w_ptR',w_ptR)
        if (isinstance(self.__tRInference['p_tR'].iloc[0],float)): self.__tRInference.insert(4,'p_tR_w',w_ptR*self.__tRInference['p_tR'])
        
        return self.__tRInference

        

    def getStructuralFormulasFromChemicalSpace(self,useProbableMolFormulas=False,useMZTarget=False,mzAccDBSearch=None):

        __tmpMzAccDBSearch=self.mzAccDBSearch
        if isinstance(mzAccDBSearch,float):
            self.mzAccDBSearch=mzAccDBSearch
        
        self.__sFormulas=None
        __mzTarget=self.mzTarget
        if not(self.__ionizedSpeciesInfo.empty):__mzTarget=self.mzTarget-self.__ionizedSpeciesInfo.mz_apported.iloc[0]

        if useProbableMolFormulas:
            if not(self.mzTarget is None):
                formulas=self.getProbableMolecularFormulas()
                dfL=[self.formulaDBsearch(molecularFormula=formula) for formula in formulas.Formula if len(self.formulaDBsearch(molecularFormula=formula))>0]
                self.__sFormulas=pd.DataFrame()
                for df in dfL:
                    self.__sFormulas=pd.concat([self.__sFormulas,df],sort=False,ignore_index=True)
                self.__sFormulas=self.__sFormulas.drop_duplicates().sort_values(['IDkey']).reset_index(drop=True)
        else:
            
            if ( (isinstance(self.formula,str)) & (not(useMZTarget)) ):
                if self.baseFormula!="":self.__sFormulas=self.formulaDBsearch(molecularFormula=self.baseFormula)
            elif ( (not(self.mzTarget is None)) ):
                self.__sFormulas=self.formulaDBsearch(float(__mzTarget))

        if not(isinstance(self.__sFormulas,type(None))): self.__sFormulas.insert(6,'mzErr_ppm',[(__mzTarget-mz)/mz*1E6 for mz in self.__sFormulas.exactMass])
        self.mzAccDBSearch=__tmpMzAccDBSearch
        
        return self.__sFormulas

  
    def plotCompoundXICWithRetentionTimeInference(self):
        self.xic=self.getChromatogram()
        self.tR=self.xic.time[self.xic.intensity.idxmax()]
        dtR=None
        UtR=0.0
        if not(self.__tRInference.empty):
            dtR=self.__tRInference.tR[0]-self.tR
            UtR=self.__tRInference.U_tR[0]

        self.plotXIC(plot2dtR=dtR,U_tR=UtR,title="")
        self.xic=pd.DataFrame()

                   
            
    def assessIsotopicPatterns(self,numOfIsoSignals=6,isoPattern=None,mzAccDBSearch=None):

        if ((isinstance(self.__sFormulas,type(None))) | (isinstance(mzAccDBSearch,float) )):
            self.__sFormulas=self.getStructuralFormulasFromChemicalSpace(useMZTarget=True,mzAccDBSearch=mzAccDBSearch)
        
        if isinstance(isoPattern,type(None)):
            isoPatternToAssess__=self.getExperimentalIsotopicPatternSignals(isPlotShown=False)
            if isinstance(isoPatternToAssess__,type(None)):    
                isoPatternToAssess__=self.theoreticalIsotopicPattern.copy()

            isoPatternToAssess__=isoPatternToAssess__[0:min(len(isoPatternToAssess__),numOfIsoSignals)].copy()
            
        else:
            isoPatternToAssess__=isoPattern[0:min(len(isoPattern),numOfIsoSignals)].copy()
            
        __numberOfIsotopicalSignals=len(isoPatternToAssess__)
        __patternsIntensityMtx=np.empty((0,__numberOfIsotopicalSignals))
        __patternsMZMtx=np.empty((0,__numberOfIsotopicalSignals))

        ionSpecies=""
        if len(self.ionizedSpecies)==2:
            ionSpecies=self.ionizedSpecies[0]

        formulasToRemove=[]
        for i,formula in enumerate(self.__sFormulas.molecularFormula):
            formula=formula+ionSpecies

            formulaData=self.getFlattenMolecularFormulaDF(formula)

            try:
                self.isoPatternsGenerator.atomicComposition=formulaData
            except:
                print(f"E.(msCompoundExplorer): Invalid formula {formula}")
                formulasToRemove.append(formula)
                continue
            ipatt=self.isoPatternsGenerator.theoreticalIsoPattern[0:__numberOfIsotopicalSignals]


            nlack=self.isoPatternsGenerator.isoPatternPeaksNumCalculator-len(ipatt)                        
            if nlack>0: 
                ipatt=pd.concat([ipatt,pd.DataFrame({'mz':ipatt.mz.iloc[-1]+range(1,nlack+1),'intensity':[0.01]*nlack,'z':[ipatt.z.iloc[0]]*nlack})],sort=False,ignore_index=True)
                
            ipatt=ipatt[0:__numberOfIsotopicalSignals]
            __patternsIntensityMtx= np.append(__patternsIntensityMtx,[ipatt.intensity.to_numpy()], axis=0)
            __patternsMZMtx= np.append(__patternsMZMtx,[ipatt.mz.to_numpy()], axis=0)


        isoSignalsToSelect=(isoPatternToAssess__.mz-isoPatternToAssess__.mz[0]).round().astype(int).to_list()
        __nTIS=__patternsMZMtx.shape[1]


        
        # ## MZ scale       
        mz_experimental=isoPatternToAssess__.mz.to_numpy()
        mz_refs=__patternsMZMtx[:,isoSignalsToSelect]
        d=1/np.einsum("ij,ij->i",mz_refs,mz_refs)
        mz_projector= np.matmul(mz_refs.T,np.diag(d)).T       
        mz_errors=np.concatenate([[self.isotopicSignals_mzErr_ppm[0]],[self.isotopicSignals_mzErr_ppm[1]]*(__nTIS-1)]).reshape(1,__nTIS)/1E6       
        mz_errors=np.array([mz_errors[0,isoSignalsToSelect]])
        mz_errors=np.matmul(np.ones((len(mz_refs),1)),mz_errors) 
        dmz_err=np.multiply(mz_refs,mz_errors)
        D_mz_proj=np.einsum("ij,ij->i",dmz_err,mz_projector)
        mzProjections=np.matmul(mz_experimental,mz_projector.T)
        Err_mz=abs((mzProjections-1)/D_mz_proj)

        ## Intensity scale
        int_experimental=isoPatternToAssess__.intensity.to_numpy()
        int_experimental=int_experimental/int_experimental.max()
        int_refs=__patternsIntensityMtx[:,isoSignalsToSelect]
        d=1/np.einsum("ij,ij->i",int_refs,int_refs)
        int_projector= np.matmul(int_refs.T,np.diag(d)).T
        int_errors=np.matrix(self.isotopicSignals_intensityErr+[self.isotopicSignals_intensityErr[-1]]*(__nTIS-len(self.isotopicSignals_intensityErr)))
        int_errors=np.array(int_errors[0,isoSignalsToSelect])
        int_errors=np.matmul(np.ones((len(int_refs),1)),int_errors) 
        dint_err=np.multiply(int_refs,int_errors)
        D_int_proj=np.einsum("ij,ij->i",dint_err,int_projector)
        intProjections=np.matmul(int_experimental,int_projector.T)
        Err_int=abs((intProjections-1)/D_int_proj)

        __combError=(Err_mz**2+Err_int**2)**0.5
        __p=[self.p_isoPattError(cErr) for cErr in __combError]
        
        if len(formulasToRemove) >0:
            self.__sFormulas=self.__sFormulas[(~self.__sFormulas["molecularFormula"].isin(formulasToRemove))].reset_index(drop=True)
        
        return pd.DataFrame({'interactionProductFormula':self.__sFormulas.molecularFormula,
                                                    'vectorialProj_intensityError':Err_int,
                                                    'vectorialProj_mzError':Err_mz,
                                                    'vectorialProj_combinedError':__combError,
                                                    'p_vectorialProj':__p,
                                                    'dExactMass_ppm':self.__sFormulas.mzErr_ppm
                                                    })

                
