import sys,os
import re,glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from scipy import special
sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from msIsotopicCluster import msIsotopicCluster
from msInsilicoFragmenter import msInsilicoFragmenter
if isinstance(os.getenv('RDKIT_PATH'),str):
    sys.path.append(os.environ['RDKIT_PATH'])
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from mordred import Calculator, descriptors

from PIL import Image
from IPython.display import display

from msMLModelsCommon import msMLModelsCommon
msMLModelsComm=msMLModelsCommon()
saltRemover = SaltRemover()

class msMolecule():
    def __init__(self):
        self.__isotopicCluster=msIsotopicCluster()
        self.__msFragmenter=msInsilicoFragmenter()
        self.__rdkit_mol=None
        self.__name=None
        self.__inchikey=None
        self.__formula=None
        self.__isValidFormula=True
        self.__isVerbose=False
        self.__baseFormula=None
        self.__baseCharge=0
        self.__baseExactMass=None
        self.__basetheoreticalIsotopicPattern=pd.DataFrame()
        self.__smiles=None
        self.__isBaseFormula=True
        self.__exactMass=pd.DataFrame() 
        self.__flattenedFormula=None
        self.__baseMolecularDescriptors=pd.DataFrame()
        self.__baseMordredMolecularDescriptors=pd.DataFrame()
        self.__amenabilityAndtR_features=pd.DataFrame()
        self.__amenabilityPrediction=None
        self.__tRPrediction=None
        self.__ionSpeciesPredictionPdP=None
        self.__ionSpeciesPrediction=None        
        self.__resultsPath="./"
        self.__formulaTransformation=None
        self.__qc_tR=0.15
       

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self,value):
        self.__name=value

    @property
    def inchikey(self):
        return self.__inchikey

    @inchikey.setter
    def inchikey(self,value):
        self.__inchikey=value
        
    @property
    def smiles(self):
        return self.__smiles

    @smiles.setter
    def smiles(self, value):

        if isinstance(value,tuple):
            smiles_,z=value
        else:
            smiles_=value
            z=0

            
        if (self.__smiles!=smiles_):
            self.__amenabilityAndtR_features=pd.DataFrame()
            self.__inchikey=None

        self.__rdkit_mol=Chem.MolFromSmiles(smiles_)
        if isinstance(self.__rdkit_mol,type(None)):
            print("E.(msMolecule): invalid smiles",smiles_)
            self.formula = None
            self.__smiles= None
            self.__inchikey= None
            self.__exactMass=pd.DataFrame()


        else:
            self.__rdkit_mol=saltRemover(self.__rdkit_mol)
            self.__smiles=Chem.MolToSmiles(self.__rdkit_mol)
            self.__inchikey=Chem.MolToInchiKey(self.__rdkit_mol)
            try:
                self.formula = (CalcMolFormula(self.__rdkit_mol,separateIsotopes=True,abbreviateHIsotopes=False),z)
            except:
                print("E.(msMolecule): invalid smiles",self.__smiles)
                self.__rdkit_mol=None
                self.formula = None
                self.__smiles= None
                self.__inchikey=None
                self.__exactMass=pd.DataFrame()

    @property
    def formula(self):
        return self.__formula

    @formula.setter
    def formula(self,value):
        
        if isinstance(value,tuple):
            formula_,z=value
        else:
            formula_=value
            z=0

        if formula_==None: return


        if "+" in formula_:

            z_=formula_.split("+")[1]
            if z_=="":
                z=1
            else:
                z=int(z_)
                
            formula_=formula_.split("+")[0]


        if "-" in formula_:

            z_=formula_.split("-")[1]
            if z_=="":
                z=-1
            else:
                z=-1*int(z_)
            
            formula_=formula_.split("-")[0]

                    
        self.__baseFormula=formula_
        self.__baseCharge=z
        self.__formula=formula_
        self.__isotopicCluster.clusterCharge=z
        self.__exactMass=self.__getExactMass()
        self.__baseExactMass=self.__exactMass
        self.__isotopicCluster.atomicComposition=self.__getAtomicComposition()
        self.__basetheoreticalIsotopicPattern=self.__isotopicCluster.theoreticalIsoPattern.copy()
        self.__baseMolecularDescriptors=pd.DataFrame()
        self.__flattenedFormula=self.__getFlattenedFormula()
        self.isoPatternFileName=formula_+"_isoPattern"

    def __setClusterCharge(self,value):
        self.__isotopicCluster.clusterCharge=value
        
    def formulawr(self,value):       
        if isinstance(value,tuple):
            formula_,z=value
        else:
            formula_=value
            z=0
        if formula_==None: return
       
        self.__baseFormula=formula_
        self.__baseCharge=z
        self.__formula=formula_
        self.__isotopicCluster.clusterCharge=z
        self.__exactMass=self.__getExactMass()
        self.__baseExactMass=self.__exactMass
        self.__isotopicCluster.atomicComposition=self.__getAtomicComposition()
        self.__basetheoreticalIsotopicPattern=self.__isotopicCluster.theoreticalIsoPattern.copy()
        self.__baseMolecularDescriptors=pd.DataFrame()
        self.__flattenedFormula=self.__getFlattenedFormula()
        self.isoPatternFileName=formula_+"_isoPattern"

    @property
    def isValidFormula(self):
        return self.__isValidFormula

    
    def __updateFormula(self,formula_,z):
        #if abs(z)>0:
        self.__formula=formula_
        self.__isotopicCluster.clusterCharge=z
        self.__exactMass=self.__getExactMass()
        self.__isotopicCluster.atomicComposition=self.__getAtomicComposition()
        self.__flattenedFormula=self.__getFlattenedFormula()
        self.isoPatternFileName=formula_+"_isoPattern"


    def resetFormula(self):
        if self.__smiles!=None:
            self.__updateFormula(CalcMolFormula(self.__rdkit_mol,separateIsotopes=True,abbreviateHIsotopes=False),self.__baseCharge)
        elif self.__baseFormula!=None:
            self.__updateFormula(self.__baseFormula,self.__baseCharge)

    @property
    def charge(self):
        return self.__isotopicCluster.clusterCharge


    @charge.setter
    def charge(self,value):
        if self.__formula!=None:
            self.formula=(self.__formula,value)

    @property
    def isotopicAbundances(self):
        return self.__isotopicCluster.isotopicAbundancesDF

                        
    @property
    def theorIsoPatternPeaksNum(self):
        return self.__isotopicCluster.isoPatternPeaksNumCalculator

    @theorIsoPatternPeaksNum.setter
    def theorIsoPatternPeaksNum(self,value):
        self.__isotopicCluster.isoPatternPeaksNumCalculator=value
                
    @property
    def isBaseFormula(self):
        return self.__isBaseFormula

    @isBaseFormula.setter
    def isBaseFormula(self,value):
        self.__isBaseFormula=value

    @property
    def baseFormula(self):
        if self.__baseFormula is None: return ""       
        return self.__baseFormula

    @property
    def baseCharge(self):
        return self.__baseCharge

    @property
    def baseExactMass(self):
        return self.__baseExactMass

    @property
    def formulaTransformation(self):
        return self.__formulaTransformation
        
    @property
    def exactMass(self):
        return self.__exactMass

    @exactMass.setter
    def exactMass(self,value):
        self.__exactMass=value


    @property
    def RT_cs(self):
        return self.__qc_tR

    @RT_cs.setter
    def RT_cs(self,value):
        self.__qc_tR=value

        
    @property
    def ionicFormula(self):
        return self.__flattenedFormula

    @property
    def atomicComposition(self):
        return self.__isotopicCluster.atomicComposition

    @property
    def theoreticalIsotopicPattern(self):
        return self.__isotopicCluster.theoreticalIsoPattern

    @property
    def baseTheoreticalIsotopicPattern(self):
        return self.__basetheoreticalIsotopicPattern

    @property
    def baseMolecularDescriptors(self):
        if self.__baseMolecularDescriptors.empty:
            self.getBaseMolecularDescriptors()
            
        return self.__baseMolecularDescriptors

    @property
    def baseMordredMolecularDescriptors(self):
        if self.__baseMordredMolecularDescriptors.empty:
            self.getBaseMordredMolecularDescriptors()
            
        return self.__baseMordredMolecularDescriptors

    @property
    def amenabilityAndtR_features(self):
        
        if self.__amenabilityAndtR_features.empty:
            self.getMolecularFeatures()
            
        return self.__amenabilityAndtR_features
   

    @property
    def physicalConstants(self):
        return self.__isotopicCluster.physicalConstants


    @property
    def isVerbose(self):
        return self.__isVerbose

    @isVerbose.setter
    def isVerbose(self,value):
        self.__isVerbose=value

    @property
    def msFragmenter(self):
        return self.__msFragmenter

    @property
    def instrumentalMethod(self):
        return msMLModelsComm.instrumentalMethod

    @instrumentalMethod.setter
    def instrumentalMethod(self,value):
        msMLModelsComm.instrumentalMethod=value

    def getSubformulas(self,formula):
        formula=formula.replace(" ","")
        subFormulas=re.split("[+-]+",formula) 
        subFormulas=[s for s in subFormulas if s]
        subFormsOpers=[-1 if s=='-' else 1 for s in re.split("[a-zA-Z0-9()\[\]]+",formula)]
        subFormsOpers=subFormsOpers[0:len(subFormsOpers)-1]
        return pd.DataFrame({'operation':subFormsOpers,'subformula':subFormulas})

    def getFormulaFactors(self,formula):
        multiplier=re.search(r"^[0-9]+",formula)
        if isinstance(multiplier,type(None)):
            multiplier=1
            formulaFactor=formula
        else:
            multiplier=int(multiplier.group(0))
            formulaFactor=re.search(r"^[0-9]+\((.*?)\)$",formula)
            if isinstance(formulaFactor,type(None)):
                formulaFactor=re.sub(r"^"+str(multiplier), "", formula)
            else:
                formulaFactor=formulaFactor.group(1)

        return pd.DataFrame({'multiplier':[multiplier],'formulaFactor':[formulaFactor]})

    def splitIsotopicFormula(self,formula):
        isotopeList=re.findall(r"[\[][0-9]*[A-Z][a-z]?[\]]",formula)
        deisotFormula = formula
        for iso_ in isotopeList: deisotFormula = deisotFormula.replace(iso_, "@")
        elementsNums=re.findall(r"[A-Z][a-z]?[0-9]*|[@]{1}[0-9]*",deisotFormula)
        elementsNums=[((re.findall(r"[A-Z][a-z]?|[@]{1}", elma))[0],int((re.findall(r"[0-9]+", elma)+["1"])[0])) for elma in elementsNums]
        ddf=pd.DataFrame({'atom':[],'numAtoms':[]})
        i=0
        for t in elementsNums:
            elm=t[0]
            if '@' in t[0]:
                elm=isotopeList[i]
                i+=1
            ddf.loc[len(ddf)] = [elm, t[1]]
        return ddf

    def getFlattenMolecularFormula(self,formula):
        sFormula=self.getFlattenMolecularFormulaDF(formula)
        sFormula=sFormula[sFormula.numAtoms!=0]
        sFormula['symOrder']=[ re.findall(r"[A-Z][a-z]?",s)[0] for s in sFormula.atom]
        sFormula.atom=[f'[{s}]' if re.match(r'^[0-9]',s) else s for s in sFormula.atom]
        sFormula=sFormula.sort_values('symOrder')
        sFormula=pd.concat([sFormula[sFormula.symOrder=='C'],sFormula[sFormula.symOrder=='H'],sFormula[(sFormula.symOrder!='C') & (sFormula.symOrder!='H')]],
                   ignore_index=True)
        sFormula.numAtoms=["" if nat==1 else str(nat) for nat in sFormula.numAtoms]
        sFormula="".join(sFormula.atom+sFormula.numAtoms)
        return sFormula


    def getFlattenMolecularFormulaDF(self,formula):
        molSubformulas=self.getSubformulas(formula)
        atomsList=pd.DataFrame()
        for subformula in molSubformulas.iterrows():
            formsAndFactors=self.getFormulaFactors(subformula[1].subformula)
            formsAndFactors['multiplier']=formsAndFactors.multiplier*subformula[1].operation
            splitedElements=self.splitIsotopicFormula(formsAndFactors.formulaFactor.iloc[0])
            splitedElements.numAtoms=splitedElements.numAtoms*formsAndFactors.multiplier.iloc[0]
            atomsList=pd.concat([atomsList,splitedElements],ignore_index=True)
        atomsList=atomsList.groupby('atom')[['numAtoms']].sum().reset_index(drop=False)
        atomsList.atom=atomsList.atom.str.replace(r"\[|\]","", regex=True)
        atomsList['atomsTypeCoded']=[(re.findall(r'[A-Z][a-z]?', at)[0]+'['+''.join(re.findall(r'[0-9]+', at))+"]").replace("[]","") for at in atomsList.atom]
        return atomsList

    def getIsotopicElementWeigth(self,element):

        atom=self.isotopicAbundances[self.isotopicAbundances.isoSymbol==element]
        if atom.empty: atom=self.isotopicAbundances[self.isotopicAbundances.allIsoSymbol==element]
        return pd.DataFrame({'mAW':[atom.AW.iloc[0]],'u_mAW':[atom.u_AW.iloc[0]],'abundance':[atom.Abundance.iloc[0]],'u_abundance':[atom.u_Abundance.iloc[0]]})    

    
    def getExactMass(self,formula,z=0):
        sFormula=self.getFlattenMolecularFormulaDF(formula)
        e_uam_mass=self.physicalConstants[self.physicalConstants.Quantity=="electron mass in u"]
        m_z=e_uam_mass.Value.iloc[0]*z
        u_m_z=e_uam_mass.Uncertainty.iloc[0]*z
        exactMW=0
        u_exactMW=0
        for atom in sFormula.iterrows():
            AW_=self.getIsotopicElementWeigth(atom[1].atom)
            exactMW=exactMW+AW_.mAW*atom[1].numAtoms
            u_exactMW=u_exactMW+(AW_.u_mAW*AW_.mAW*atom[1].numAtoms)**2

        exactMW=np.sign(exactMW)*(abs(exactMW)-m_z)
        u_exactMW=np.sqrt(u_exactMW+u_m_z**2)
        mz=np.nan
        if( z!=0): mz=exactMW/abs(z)
        return pd.DataFrame({'exactMW':exactMW,'u_exactMW':u_exactMW,'mz':mz})
       
    
    def plotIsotopicPattern(self,isPlotSaved=False,isPlotShown=True):
        #self.__isotopicCluster.plotIsotopicPattern(isPlotSaved,isPlotShown)
        self.__isotopicCluster.plotIsotopicPattern()
               
    def __getExactMass(self):
        if self.__formula!=None:
            return self.getExactMass(self.__formula,z=int(self.__isotopicCluster.clusterCharge))

       
    def __getAtomicComposition(self):
        if self.__formula==None: self.__isValidFormula=False
        if self.__formula!=None:
            df=self.getFlattenMolecularFormulaDF(self.__formula)
            self.__isValidFormula=True
            if int(df.numAtoms.min())<0:
                self.__isValidFormula=False
                if self.__isVerbose:
                    print('W.(msMolecule): invalid formula')

            return df
       

    def __getFlattenedFormula(self):
        if self.__formula!=None:
            return self.getFlattenMolecularFormula(self.__formula)

    def transformFormula(self,formulaTransformation=None,zForTransformation=np.nan):
                       
        if self.__formula!=None:

            if (formulaTransformation=="") | (str(formulaTransformation)=="nan"): return
            
            __zForTransformation=zForTransformation
            self.__formulaTransformation=formulaTransformation

            if (self.__formulaTransformation != None):

                if (not(("+" in self.__formulaTransformation[0]) | ("-" in self.__formulaTransformation[0]))):
                    self.__formulaTransformation="+"+formulaTransformation

                if ("[M]" in self.__formulaTransformation ):

                    charge_str="0+"+"".join(re.findall("[+-]?[0-9]?\[M",formulaTransformation))
                    replacements=[("--","-"),("++","+"),("-[","-"),("+[","+"),("[","*"),("M","1")]
                    for strR in replacements: charge_str=charge_str.replace(strR[0],strR[1])                   
                    __zForTransformation=__zForTransformation+self.__baseCharge*eval(charge_str)
                    self.__formulaTransformation=self.__formulaTransformation.replace("[M]",f"({self.baseFormula})")

                    
            if self.__isBaseFormula:

                if np.isnan(__zForTransformation):
                    self.__updateFormula(self.__baseFormula+self.__formulaTransformation,self.__baseCharge)
                else:
                    self.__updateFormula(self.__baseFormula+self.__formulaTransformation,self.__baseCharge+__zForTransformation)

            else:
                if np.isnan(__zForTransformation):
                    z=self.clusterCharge
                    self.__updateFormula(self.__formula+self.__formulaTransformation,z)
                else:
                    self.__updateFormula(self.__formula+self.__formulaTransformation,self.__baseCharge+__zForTransformation)
                   

    def show2DMolecularStructure(self, isSaved=False,extsmiles=None,name=None):

        __rdkit_mol=self.__rdkit_mol
        if extsmiles!=None: __rdkit_mol=Chem.MolFromSmiles(extsmiles)
       
        if isSaved:
            if isinstance(name,str):

                if name!="":
                    fileName=os.path.join(self.__resultsPath,f"{name}.png")
                else:
                    fileName=os.path.join(self.__resultsPath,f"{self.__inchikey}.png")
            else:
                fileName=os.path.join(self.__resultsPath,f"{self.__baseFormula}.png")

            Draw.MolToFile(__rdkit_mol,
                            fileName,
                            size=(300, 300),
                            kekulize=True,
                            wedgeBonds=True,
                            fitImage=True)

            
            molImg = Image.open(fileName)
            display(molImg)

            if not(isinstance(name,str)): os.remove(fileName)
        else:
            molDraw=Draw.MolToImage(__rdkit_mol,
                                    size=(300, 300),
                                    kekulize=True,
                                    wedgeBonds=True,
                                    fitImage=True,
                                    canvas=None)
            molDraw.show()



    def showMolecularStructure(self,extsmiles=None,legend=""):

        __rdkit_mol=self.__rdkit_mol
        __legend=legend
        if self.__name!=None: __legend=self.__name
        if legend!="": __legend=legend
        if extsmiles!=None:
            __rdkit_mol=Chem.MolFromSmiles(extsmiles)
            __legend=legend

        molc=Draw.MolDraw2DCairo(300,300)
        molc.DrawMolecule(__rdkit_mol,legend=__legend)
        molc.FinishDrawing()
        bio = BytesIO(molc.GetDrawingText())
        return Image.open(bio)
        

    def rdkitMolWithoutIsotopes(self,rdkit_mol):
        atom_data = [(atom, atom.GetIsotope()) for atom in rdkit_mol.GetAtoms()]
        for atom, isotope in atom_data:
            if isotope: atom.SetIsotope(0)
        return Chem.MolFromSmiles(Chem.MolToSmiles(rdkit_mol))
    
            
    def getBaseMolecularDescriptors(self):

        self.__baseMolecularDescriptors=pd.DataFrame()
        if self.__rdkit_mol!=None:
            self.__baseMolecularDescriptors=pd.DataFrame(Descriptors.CalcMolDescriptors(self.rdkitMolWithoutIsotopes(self.__rdkit_mol)),index=[0])


    def getBaseMordredMolecularDescriptors(self):

        self.__baseMordredMolecularDescriptors=pd.DataFrame()
        if self.__rdkit_mol!=None:

            calc = Calculator(descriptors, ignore_3D=True)            
            self.__baseMordredMolecularDescriptors=pd.DataFrame(calc(self.rdkitMolWithoutIsotopes(self.__rdkit_mol)).asdict(),index=[0]).astype(float)
        
    def getMolecularFeatures(self):

        if self.__rdkit_mol!=None:
            if self.__amenabilityAndtR_features.empty:
                self.getBaseMolecularDescriptors()
                self.__amenabilityAndtR_features=self.baseMolecularDescriptors
                self.getBaseMordredMolecularDescriptors()
                self.__amenabilityAndtR_features['MID_N']=self.__baseMordredMolecularDescriptors.MID_N.iloc[0]
                self.__amenabilityAndtR_features['MID_O']=self.__baseMordredMolecularDescriptors.MID_O.iloc[0]
                self.__amenabilityAndtR_features['MID_H']=self.__baseMordredMolecularDescriptors.MID_h.iloc[0]
                self.__amenabilityAndtR_features['MID_X']=self.__baseMordredMolecularDescriptors.MID_X.iloc[0]
                self.__amenabilityAndtR_features['apol']=self.__baseMordredMolecularDescriptors.apol.iloc[0]
                self.__amenabilityAndtR_features['bpol']=self.__baseMordredMolecularDescriptors.bpol.iloc[0]
                self.__amenabilityAndtR_features['LogS']=self.__baseMordredMolecularDescriptors.FilterItLogS.iloc[0]
                if self.__amenabilityAndtR_features.isin([np.inf, -np.inf]).iloc[0].any(): self.__amenabilityAndtR_features=pd.DataFrame()

            return self.__amenabilityAndtR_features


    def getRetentionTimeInference(self,tR_exp=None):
        tR_predictorFeatures=self.getMolecularFeatures()
        if tR_predictorFeatures.empty: return pd.DataFrame()
        return msMLModelsComm.getRetentionTimePrediction(tR_predictorFeatures,tR_exp)

    def getRetentionTimeAssessment(self,tR_exp,tR_ref,U_tR):
        #s_z=np.sqrt(2*(s_tR**2)) # s_tR=0.01
        s_z=0.01414213562373095
        p=0.5*(special.erf(((tR_ref-tR_exp)+self.__qc_tR)/(np.sqrt(2)*s_z))-special.erf(((tR_ref-tR_exp)-self.__qc_tR)/(np.sqrt(2)*s_z)))#/den
        return pd.DataFrame({'tR':[tR_ref],'U_tR':U_tR,'p_tR':[p]})
    

    def getAmenabilityInference(self):

        am_predictorFeatures=self.getMolecularFeatures()
        if am_predictorFeatures.empty:return pd.DataFrame()
        self.__amenabilityPrediction=msMLModelsComm.getAmenabilityInference(am_predictorFeatures)
        return self.__amenabilityPrediction

    def getIonizationSpeciesPdPInference(self):
        if_predictorFeatures=self.getMolecularFeatures()
        if if_predictorFeatures.empty:return pd.DataFrame()
        self.__ionSpeciesPredictionPdP=msMLModelsComm.getIonizationSpeciesPdPInference(if_predictorFeatures)
        return self.__ionSpeciesPredictionPdP

    def getIonizationSpeciesInference(self):
        if_predictorFeatures=self.getMolecularFeatures()
        if if_predictorFeatures.empty:return pd.DataFrame()
        self.__ionSpeciesPrediction=msMLModelsComm.getIonizationSpeciesInference(if_predictorFeatures)
        return self.__ionSpeciesPrediction
    

