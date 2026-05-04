import sys,os
import pandas as pd
import numpy as np

sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from msMolecule import msMolecule
from  msCommonStructureData import msCommonStructureData
msCommDS=msCommonStructureData()

class msInteractionProduct(msMolecule):
    def __init__(self):
        msMolecule.__init__(self)

        ## Interactions
        self.__ionsSpeciesPolarity="+"
        self.__adductSpecies={} 
        self.__replacements={} 
        self.__losses={} 
        self.__interactionProducts= self.__adductSpecies | self.__replacements | self.__losses
        self.__currInteracProdIdx=0
        self.__currProductInteration=""
        self.__currProductInterationCharge=0
        self.__ionSpeciesLib=pd.read_csv(os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],"msData/ionizedSpecies.tsv"),sep="\t")[['formula','z']]
        self.__iformsExactMass=pd.DataFrame()
        self.__commonIonSpecies=pd.read_csv(os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],"msData/commonIonizedSpecies.tsv"),sep="\t")


    def __iter__(self):
        return self

    def __next__(self):
        if self.__currInteracProdIdx < len(self.__interactionProducts):

            if self.isVerbose:
                print(f"\nI.(msInteractionProduct): Ion: ({self.__currInteracProdIdx}) \n")

            try:
                self.__getInteration(self.__currInteracProdIdx)
            except:
                print(f"E.(msInteractionProduct): Formula: {self.formula}\n")
                self.__currInteracProdIdx+=1
                return None

            
            if self.isVerbose:
                print(f"I.(msInteractionProduct): Formula: {self.formula}\n")


            self.__currInteracProdIdx+=1

            return self.formula
        else:
            self.__currInteracProdIdx=0
            raise StopIteration
        
    def __getInteration(self,interactionIdx):

        self.__currProductInteration=list(self.__interactionProducts)[interactionIdx]
        self.__currProductInterationCharge=self.__interactionProducts[self.__currProductInteration]
        self.transformFormula(self.__currProductInteration,self.__currProductInterationCharge)


    def __upddateInteractions(self):
        self.__interactionProducts= self.__adductSpecies | self.__replacements | self.__losses

    @property
    def currentIonSpecies(self):
        return (self.__currProductInteration,self.__currProductInterationCharge)
        
        
    @property
    def ionSpeciesExactMass(self):
        if self.__iformsExactMass.empty:
            self.getInteractionFormulasExactMassDiff()
            
        return self.__iformsExactMass

    @property
    def ionsSpeciesPolarity(self):
        return self.__ionsSpeciesPolarity

    @ionsSpeciesPolarity.setter
    def ionsSpeciesPolarity(self,value):
        self.__ionsSpeciesPolarity=value
            
    @property
    def adductSpecies(self):
        return pd.DataFrame.from_dict(self.__adductSpecies,orient="index").reset_index().rename(columns={'index':'adductFormula',0:'charge'})


    @adductSpecies.setter
    def adductSpecies(self,value):

        if( isinstance(value,dict) ):
            self.__adductSpecies.update(value)
        elif isinstance(value,str):
            values_=self.__readInteractingFormulas(value,typeIF="adductSpecies")
            self.__adductSpecies.update(values_)
        self.__upddateInteractions()

    @property
    def losses(self):
        return pd.DataFrame.from_dict(self.__losses,orient="index").reset_index().rename(columns={'index':'lossFormula',0:'charge'})

    @losses.setter
    def losses(self,value):

        if( isinstance(value,dict) ):
            self.__losses.update(value)
        elif isinstance(value,str):
            self.__losses.update(self.__readInteractingFormulas(value,typeIF="losses"))

        self.__upddateInteractions()

    @property
    def replacements(self):
        return pd.DataFrame.from_dict(self.__replacements,orient="index").reset_index().rename(columns={'index':'replacementFormula',0:'charge'})

    @replacements.setter
    def replacements(self,value):

        if( isinstance(value,dict) ):
            self.__replacements.update(value)
        elif isinstance(value,str):
            self.__replacements.update(self.__readInteractingFormulas(value,typeIF="replacements"))

        self.__upddateInteractions()


    @property
    def commonIonSpecies(self):
        return self.__commonIonSpecies

    @commonIonSpecies.setter
    def commonIonSpecies(self,file):
        self.__commonIonSpecies=pd.read_csv(file,sep="\t")    
        
    @property
    def interactionProducts(self):

        ipdf=pd.DataFrame.from_dict(self.__interactionProducts,orient="index").reset_index().rename(columns={'index':'interactionFormula',0:'charge'})
        ipdf.interactionFormula=self.baseFormula+ipdf.interactionFormula
       
        return ipdf

    @property
    def interactionReactants(self):

        ipdf=pd.DataFrame.from_dict(self.__interactionProducts,orient="index").reset_index().rename(columns={'index':'interactionReactant',0:'charge'})
       
        return ipdf

    

    @property
    def allInteractionFormulas(self):      
        return pd.DataFrame.from_dict(self.__interactionProducts,orient="index").reset_index().rename(columns={'index':'interactionFormula',0:'charge'})

    @property
    def ionSpeciesLibrary(self):
        return self.__ionSpeciesLib


    @ionSpeciesLibrary.setter
    def ionSpeciesLibrary(self,file):
        self.__ionSpeciesLib=pd.read_csv(file,sep="\t")[['formula','z']]
    

    @property
    def numOfInteractionProducts(self):
        return len(self.__interactionProducts)

    @interactionProducts.setter
    def interactionProducts(self,value):
        self.__adductSpecies={}
        self.__replacements={}
        self.__losses={}
        self.__interactionProducts={}

       
        if( isinstance(value,dict) ):
            self.__interactionProducts.update(value)

        elif isinstance(value,str):
            self.__losses.update(self.__readInteractingFormulas(value,typeIF="losses"))
            self.__interactionProducts.update(self.__readInteractingFormulas(value,typeIF="replacements"))
            self.__adductSpecies.update(self.__readInteractingFormulas(value,typeIF="adductSpecies"))           
            self.__upddateInteractions()

        elif isinstance(value,list):
            self.__interactionProducts={__ionSpecies:self.__ionSpeciesLib[self.__ionSpeciesLib.formula==__ionSpecies]['z'].iloc[0] for __ionSpecies in value}

            
            
        
    def __readInteractingFormulas(self,file,typeIF="adductSpecies"):

        if os.path.exists(file):   
            interactingFormulasData=pd.read_csv(file,sep=",")
            return msCommDS.pdToDict(interactingFormulasData[interactingFormulasData['type']==typeIF],['interactingFormula','charge'])
                            

    def removeAdductSpecies(self,value):
        self.__adductSpecies.pop(value)

    @property
    def interactionProductIndex(self):
        return self.__currInteracProdIdx
        
    @interactionProductIndex.setter
    def interactionProductIndex(self,value):
            self.__currInteracProdIdx=value%len(self.__interactionProducts)
            self.__getInteration(self.__currInteracProdIdx)


    def getInteractionFormulasExactMassDiff(self):

        eLost=self.physicalConstants[self.physicalConstants.Quantity=="electron mass in u"]
        self.__iformsExactMass=pd.DataFrame()
        k=0
        
        for i,formula in enumerate(self.__interactionProducts):
            k+=1
            if (formula!="+") & (formula!="-"):

                formula_=formula
                if self.baseFormula!="": formula_=formula.replace("[M]","("+self.baseFormula+")")
                if not('[M]' in formula_):
                    em=self.getExactMass(formula_,0)
                    if not(em.empty):
                        em.exactMW.iloc[0]=em.exactMW.iloc[0]-1.0*int(self.__interactionProducts[formula])*eLost.Value
                        em=pd.DataFrame({'formula':[formula],'z':[int(self.__interactionProducts[formula])]}).join(em)
                        self.__iformsExactMass=pd.concat([self.__iformsExactMass,em],sort=False,ignore_index=True)
                
            else:
                k=1.0
                if formula=="+":
                    k=-1.0               
                    
                em=pd.DataFrame({'exactMW':[k*eLost.Value.iloc[0]],'u_exactMW':[eLost.Uncertainty.iloc[0]],'mz':[k*eLost.Value.iloc[0]]})
                em=pd.DataFrame({'formula':[formula],'z':[int(self.__interactionProducts[formula])]}).join(em)
                
                self.__iformsExactMass=pd.concat([self.__iformsExactMass,em],sort=False,ignore_index=True)
                self.__iformsExactMass=self.__iformsExactMass.sort_values('exactMW').reset_index(drop=True).drop(columns=['mz'])
        
        return self.__iformsExactMass


    def getIonProductsExactMass(self):

        eLost=self.physicalConstants[self.physicalConstants.Quantity=="electron mass in u"]
        self.__ionProductsExactMass=pd.DataFrame()
        k=0
        
        for i,formula in enumerate(self.__interactionProducts):
            k+=1
            if (formula!="+") & (formula!="-"):

                formula_=self.baseFormula+formula
                if self.baseFormula!="": formula_=self.baseFormula+formula.replace("[M]","("+self.baseFormula+")")
                if not('[M]' in formula_):
                    em=self.getExactMass(formula_,self.baseCharge+int(self.__interactionProducts[formula]))
                    if not(em.empty):
                        em=pd.DataFrame({'formula':[formula_],'z':[self.baseCharge+int(self.__interactionProducts[formula])]}).join(em)
                        self.__ionProductsExactMass=pd.concat([self.__ionProductsExactMass,em],sort=False,ignore_index=True)
                
            else:
                em=self.getExactMass(self.baseFormula,self.baseCharge+int(self.__interactionProducts[formula]))                    
                em=pd.DataFrame({'formula':[self.baseFormula],'z':[self.baseCharge+int(self.__interactionProducts[formula])]}).join(em)
                
                self.__ionProductsExactMass=pd.concat([self.__ionProductsExactMass,em],sort=False,ignore_index=True)
                self.__ionProductsExactMass=self.__ionProductsExactMass.sort_values('exactMW').reset_index(drop=True).drop(columns=['mz'])
        
        return self.__ionProductsExactMass

        

    def transformFormulaInverse(self,formulaTransformation,zForTransformation):
        self.transformFormula(formulaTransformation.replace("-","--").replace("+","-").replace("--","+"),
                              self.baseCharge-1.0*zForTransformation)
        return self.reduceFormula
