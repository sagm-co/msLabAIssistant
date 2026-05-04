import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
if not(isinstance(os.getenv('RDKIT_PATH'),type(None))):
    sys.path.append(os.environ['RDKIT_PATH'])
from rdkit import Chem

sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from msInteractionProduct import msInteractionProduct

class msMultiCompounds():
    def __init__(self):
        self.interactionFormulas=msInteractionProduct() 
        
        ## Formulas
        self.__smilesFormulas=pd.DataFrame() 
        self.__ionSpecies4SmilesFormulas=pd.DataFrame()

        ## System state
        self.__currentCompound=pd.DataFrame()
        self.__currentIonSpecies={}
        self.__currCompoundIdx=0
        self.__compoundsPolarity="+"


    def __iter__(self):
        return self

    def __next__(self):
        if not(self.__smilesFormulas.empty):
            if self.__currCompoundIdx < len(self.__smilesFormulas):

                self.__currentCompound=self.__smilesFormulas.iloc[self.__currCompoundIdx].copy()
                self.__updateCurrentCompound()               
                self.__currCompoundIdx+=1
                return self.__currentCompound 
            else:
                self.__currCompoundIdx=0
                raise StopIteration

    @property
    def numOfSmilesFormulas(self):
        return len(self.__smilesFormulas)

    @property
    def ionSpecies(self):
        return self.__ionSpecies4SmilesFormulas

            
    @property
    def smilesFormulas(self):
        return self.__smilesFormulas



    @smilesFormulas.setter
    def smilesFormulas(self,value):
        self.__ionSpecies4SmilesFormulas=pd.DataFrame()
        self.__ionSpecies4SmilesFormulasProbs=pd.DataFrame()
        
        if isinstance(value,pd.core.frame.DataFrame):
            self.__smilesFormulas=value.copy().reset_index(drop=True)
            
        elif isinstance(value,str):
            if os.path.exists(value):
                self.__smilesFormulas=pd.read_csv(value,sep="\t")
            else:
                print("W.(msMultiCompounds): The formulas file does not exit")                

        if not(self.__smilesFormulas.empty):
            if 'ionizedSpecies' in self.__smilesFormulas.columns:
                self.__ionSpecies4SmilesFormulas=self.__smilesFormulas.ionizedSpecies.dropna()
                if len(self.__ionSpecies4SmilesFormulas)==len(self.__smilesFormulas):
                    self.__ionSpecies4SmilesFormulas=self.__smilesFormulas[['ionizedSpecies']].copy()
                    self.__ionSpecies4SmilesFormulas['p']=[[1.0]*len(ib.split(";")) for ib in self.__ionSpecies4SmilesFormulas.ionizedSpecies]
                
                    
                else:
                    self.__ionSpecies4SmilesFormulas=pd.DataFrame(columns=['ionizedSpecies','p'])            
                    for comp in self.__smilesFormulas.iterrows():

                        if str(comp[1].ionizedSpecies)=="nan":
                            __inferedIonsData=self.__ionSpeciesInference(comp[1].smiles)
                            __inferedIons=";".join(__inferedIonsData[0])
                            p__inferedIons=__inferedIonsData[1]
                        else:
                            __inferedIons=comp[1].ionizedSpecies
                            p__inferedIons=[1.0]*len(__inferedIons.split(";"))
                            
                        self.__ionSpecies4SmilesFormulas=pd.concat([self.__ionSpecies4SmilesFormulas,
                                                                        pd.DataFrame({"ionizedSpecies":[__inferedIons],'p':[p__inferedIons]})],sort=False,ignore_index=True)

            if self.__ionSpecies4SmilesFormulas.empty:
                self.__ionSpecies4SmilesFormulas=pd.DataFrame(columns=['ionizedSpecies','p'])            
                for smiles in self.__smilesFormulas.smiles:
                    __inferedIonsData=self.__ionSpeciesInference(smiles)
                    __inferedIons=";".join(__inferedIonsData[0])
                    self.__ionSpecies4SmilesFormulas=pd.concat([self.__ionSpecies4SmilesFormulas,
                                                                pd.DataFrame({"ionizedSpecies":[__inferedIons],'p':[__inferedIonsData[1]]})],sort=False,ignore_index=True)                    
                    
            self.__smilesFormulas=self.__smilesFormulas[['name','inchikey','smiles']]
            self.__smilesFormulas.inchikey=self.__smilesFormulas.inchikey.astype(str)
            
            for idx,smFormula in enumerate(self.__smilesFormulas.smiles):
                self.__smilesFormulas.at[idx,'smiles']=Chem.MolToSmiles(Chem.MolFromSmiles(smFormula))
                if self.__smilesFormulas.inchikey.iloc[idx]=='nan':
                    self.__smilesFormulas.at[idx,'inchikey']=Chem.MolToInchiKey(Chem.MolFromSmiles(smFormula))

            self.currentCompoundIndex=0
            self.compoundCounter=0

    @property
    def totalIons(self):
        if not(self.__ionSpecies4SmilesFormulas.empty):
            return len(";".join(self.__ionSpecies4SmilesFormulas.ionizedSpecies.to_list()).split(";"))       
            
    @property
    def currentCompound(self):
        return self.__currentCompound
            
    @property
    def currentCompoundIndex(self):
        return self.__currCompoundIdx
        
    @currentCompoundIndex.setter
    def currentCompoundIndex(self,value):
        if not(self.__smilesFormulas.empty):
            self.__currCompoundIdx=value%len(self.__smilesFormulas)
            self.__currentCompound=self.__smilesFormulas.iloc[self.__currCompoundIdx]                
            self.__updateCurrentCompound()
            return

    @property
    def compoundsPolarity(self):
        return self.__compoundsPolarity

    @compoundsPolarity.setter
    def compoundsPolarity(self,value):
        self.__compoundsPolarity=value


    @property
    def currentIonizedSpeciesProb(self):
        __probs=self.__ionSpecies4SmilesFormulas.iloc[self.__currCompoundIdx].p
        return __probs

        
    def __ionSpeciesInference(self,smiles):
        self.interactionFormulas.smiles=smiles
        __ifInferenceA=self.interactionFormulas.getIonizationSpeciesInference()
        __inferedIons=self.interactionFormulas.commonIonSpecies.copy()
        __inferedIons=__inferedIons.set_index('ionSpecies').join(__ifInferenceA.set_index('ionSpecies')).dropna()[['polfreq','p','ionicProduct','polarity']]
        __inferedIons['p_prod']=__inferedIons.polfreq*__inferedIons.p
        __inferedIons=__inferedIons.sort_values(['p_prod'],ascending=False)

        __ifInference=__inferedIons[(__inferedIons.p_prod>=1) & __inferedIons.ionicProduct]
        __inferedIons=__ifInference.index.to_list()
        __ifInference=__ifInference['p'].to_list()
        if len(__inferedIons)==0:
            __inferedIons=['+H','-H']
            __ifInference=__ifInferenceA.set_index('ionSpecies').loc[__inferedIons]['p'].to_list()

        return (__inferedIons,__ifInference)
       

    def __updateCurrentCompound(self):
        self.interactionFormulas.name=self.__currentCompound['name']
        self.interactionFormulas.inchikey=self.__currentCompound.inchikey
        self.interactionFormulas.smiles=self.__currentCompound.smiles 

        if not(self.__ionSpecies4SmilesFormulas.empty):
            self.interactionFormulas.interactionProducts=self.__ionSpecies4SmilesFormulas.iloc[self.__currCompoundIdx].ionizedSpecies.split(";")
        else:
            None 

        self.interactionFormulas.interactionProductIndex=0 
        
        return

       
