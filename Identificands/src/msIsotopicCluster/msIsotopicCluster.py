import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],'External/IsotopicPatterns/'))
from brainpyIsoPatt import isotopic_variants


class msIsotopicCluster():
    def __init__(self):
        self.__atomicComposition=pd.DataFrame()
        self.__formula=""
        self.__clusterCharge=0.0
       
        ## Isotopic pattern
        self.__theoreticalIsotopicPattern=pd.DataFrame() 
        self.__nPeaksIsoPatt=6
        self.__nPeaksIsoPattCalculator=6
        self.__isoPatternFileName="isoPattern"
        self.__resultsPath="./"

        ## Isotopic generator info
        self.__physicalConstantsFile=os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],"msData/physicalConstants.dat")
        self.__physicalConstants=pd.read_csv(self.__physicalConstantsFile,comment="#",sep=";")
        self.__chargeCarrierMass=self.__physicalConstants[self.__physicalConstants.Quantity=="electron mass in u"].Value.iloc[0]
        self.__chargeCarrierCharge=-1.0
        self.__chargeCarrier=self.__chargeCarrierCharge*self.__chargeCarrierMass 
        self.__isotopicAbundancesFile=os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],"msData/isotopicAbundancesExplicit.dat")
        self.__setAbundacesLibrary(self.__isotopicAbundancesFile)

        
    @property
    def atomicComposition(self):
        return self.__atomicComposition

    @atomicComposition.setter
    def atomicComposition(self,value):
        self.__atomicComposition=value
        self.__theoreticalIsotopicPattern=self.getTheoreticaIsotopicPattern()
        
    @property
    def clusterCharge(self):
        return self.__clusterCharge

    @clusterCharge.setter 
    def clusterCharge(self,value,isFormulaUpdated=False):
        self.__clusterCharge=value
        if isFormulaUpdated:
            self.__theoreticalIsotopicPattern=self.getTheoreticaIsotopicPattern()

    @property
    def theoreticalIsoPattern(self):
        return self.__theoreticalIsotopicPattern

        
    @property
    def isoPatternFileName(self):
        return self.__isoPatternFileName

    @isoPatternFileName.setter
    def isoPatternFileName(self,value):
        self.__isoPatternFileName=value

    @property
    def chargeCarrier(self):
        return self.__chargeCarrier

    @property
    def chargeCarrierMass(self):
        return self.__chargeCarrierMass
    
    @chargeCarrier.setter
    def chargeCarrier(self,value):
        if len(value)!=2:
            print("I.(msIsotopicCluster): Set as (mass,charge) for the carrier")
            return
        mass,charge=value
        self.__chargeCarrierMass=mass
        self.__chargeCarrierCharge=charge
        self.__chargeCarrier=mass*charge
        if not(self.__atomicComposition.empty):
            self.__theoreticalIsotopicPattern=self.getTheoreticaIsotopicPattern()


            
    @property
    def isotopicAbundancesFile(self):
        return self.__isotopicAbundancesFile
    

    @isotopicAbundancesFile.setter
    def isotopicAbundancesFile(self,file):
        self.__isotopicAbundancesFile=file
        self.__setAbundacesLibrary(self.__isotopicAbundancesFile)
        if not(self.__atomicComposition.empty):
            self.__theoreticalIsotopicPattern=self.getTheoreticaIsotopicPattern()
    
    @property
    def isoPatternPeaksNum(self):
        return self.__nPeaksIsoPatt    
           
    @isoPatternPeaksNum.setter
    def isoPatternPeaksNum(self,value):
        self.__nPeaksIsoPatt=value


    @property
    def isoPatternPeaksNumCalculator(self):
        return self.__nPeaksIsoPattCalculator    

    @isoPatternPeaksNumCalculator.setter
    def isoPatternPeaksNumCalculator(self,value):
        self.__nPeaksIsoPattCalculator=value        


    @property
    def isotopicAbundances(self):
        return self.__isotopicAbundances

    @property
    def isotopicAbundancesDF(self):
        return self.__isotopicAbundancesDF
    
    @property
    def physicalConstants(self):
        return self.__physicalConstants
  
    def __setAbundacesLibrary(self,file):

        if os.path.exists(file):
            self.__isotopicAbundancesDF=pd.read_csv(file,sep=";")
            isoData=self.__isotopicAbundancesDF[['Symbol','MN','AW','Abundance']]
            isolib={}
            for elm in isoData.Symbol.unique():
                elmData=isoData[isoData.Symbol==elm].reset_index(drop=True)
                
                abnElm=lambda i :  (elmData.MN.iloc[i],(elmData.AW.iloc[i],elmData.Abundance.iloc[i]))
                maxAbd=elmData.iloc[elmData.Abundance.idxmax()]
                elmDict=[(0,(maxAbd.iloc[2],1.0))]+ list(map(abnElm,range(len(elmData.MN))))
                isolib[elm]=dict(elmDict)

            self.__isotopicAbundances=isolib
            

                   
    def getTheoreticaIsotopicPattern(self):
        
        if not(self.__atomicComposition.empty):
            
            if int(self.__atomicComposition.numAtoms.min())<0:
                self.isotopicPattUpperBound=0
                return pd.DataFrame()

            
            molComp={atom_:int(self.__atomicComposition.numAtoms[i]) for i,atom_ in enumerate(self.__atomicComposition.atomsTypeCoded)}                
            isoPattern = isotopic_variants(molComp,
                                           self.__chargeCarrier,
                                           self.__isotopicAbundances,
                                           npeaks=self.__nPeaksIsoPattCalculator,
                                           charge=self.__clusterCharge) 

            selections=[True]+list((isoPattern.mz[1:].reset_index(drop=True)-isoPattern.mz[0:-1])<2.5) 
            isoPattern = isoPattern[selections]
            isoSignalsToSelect=(isoPattern.mz-isoPattern.mz[0]).round().astype(int).to_list()
            isoPattern.index=isoSignalsToSelect
            for i in set(np.arange(max(isoSignalsToSelect)))-set(isoSignalsToSelect):
                isoPattern.loc[i]=[isoPattern.iloc[0].mz+i,0.0,isoPattern.iloc[0].z]
            isoPattern=isoPattern.sort_index().reset_index(drop=True)

            
            self.__nPeaksIsoPatt=len(isoPattern)            
            self.isotopicPattUpperBound=isoPattern.mz.iloc[-1]
            
                    
            return isoPattern
        
           
    def plotIsotopicPattern(self):
        if self.__theoreticalIsotopicPattern.empty:
            return
        
        plt.rcdefaults()
        plt.clf()
        markerline, stemlines, baseline=plt.stem(self.__theoreticalIsotopicPattern.mz,
                                                 self.__theoreticalIsotopicPattern.intensity,
                                                 markerfmt='none',linefmt='y-',label='Calculated')
        stemlines.set_alpha(0.5)
        baseline.set_color('black')
        baseline.set_linewidth(0.1)
        plt.xlabel('m/z')
        plt.ylabel('Relative Intensity')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.rcdefaults()
        plt.clf()


