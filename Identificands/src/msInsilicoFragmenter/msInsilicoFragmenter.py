# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import os
import subprocess
from datetime import  datetime
import pandas as pd
#pd.options.mode.copy_on_write = True
import numpy as np
import matplotlib.pyplot as plt
import io
import re
from PIL import Image
from IPython.display import display

sys.path.append(os.environ['IDENTIFICANDS_BASEPATH'])
from wrapers.fragmenters import fragmentersCL

if not(isinstance(os.getenv('RDKIT_PATH'),type(None))):
    sys.path.append(os.environ['RDKIT_PATH'])
    
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


class msInsilicoFragmenter(fragmentersCL):

    def __init__(self):
        fragmentersCL.__init__(self)
        self.__ionizationPolarity='+'
        self.__currIonizationPolarity=""
        self.__maxDeepFragmentation=1
        self.__ionSpecies={"+H":1,"+NH4":18,"+Na":23,"+K":39, "+CH3OH+H":33, "+CH3CN+H":42, "+CH3CN+Na":64,"+2(CH3CN)+H":83,"-H":-1,"+Cl":35,"+HCO2":45,"+CH3CO2":59}
        self.__metFragIonSpecies=1 # 1: [M+H]+, 18 [M+NH4]+, 23: [M+Na]+, 39: [M+K]+, 33: [M+CH3OH+H]+, 42: [M+ACN+H]+, 64: [M+ACN+Na]+,83: [M+2ACN+H]+, -1: [M-H]-, 35: [M+Cl]-, 45: [M+HCOO]-, 59: [M+CH3COO]-
        self.__outputPath="./"
        self.__outputNameFrag=""
        self.__outputNamePred=""
        self.__outputNameAnnot=""
        self.__smiles=""
        self.__cmdArguments=""
        self.__removeOutputFile=True

        #dataframe: mz,smiles, intensity (for prediction), energy scale
        self.__msSpectrumData=pd.DataFrame()
        self.__msSpectrum4Annotation=pd.DataFrame()
        self.__annotationData=pd.DataFrame()
        self.__fragmentsProbs=pd.DataFrame()

        # RDkit
        self.__rdkit_mol=""
        
        
    def __setFragmenterArgs(self):

        __ionizationPolarity=self.ionizationPolarity

        if not(isinstance(self.ionizationPolarity,str)):
            __ionizationPolarity="+"
            if self.ionizationPolarity<0:
                __ionizationPolarity="-"
           

    @property
    def msSpectrum4Annotation(self):
        return self.__msSpectrum4Annotation
          
    @msSpectrum4Annotation.setter
    def msSpectrum4Annotation(self,value):
        self.__msSpectrum4Annotation=value

    @property
    def msAnnotationData(self):
        return self.__annotationData

    @msAnnotationData.setter
    def msAnnotationData(self,value):
        self.__annotationData=value
    
    @property
    def metfragFragmentsProbsType(self):
        return self.__fragmentsProbs    

    @property
    def metFragIonSpecies(self):
        return self.__metFragIonSpecies

    @metFragIonSpecies.setter
    def metFragIonSpecies(self,value):
        if isinstance(value,int):
            self.__metFragIonSpecies=value
        elif isinstance(value,str):
            if value in self.__ionSpecies.keys():
                self.__metFragIonSpecies=self.__ionSpecies[value]
            else:
                print("W.(msInsilicoFragmenter): invalid ionized species for Metfrag setting")
        
    @property
    def fragOutputFile(self):
        return f"{self.__outputPath}/{self.__outputNameFrag}.dat"


    @property
    def msPredOutputFile(self):
            return f"{self.__outputPath}/{self.__outputNamePred}.dat"

    @property
    def removeOutputFile(self):
        return self.__removeOutputFile

    @removeOutputFile.setter
    def removeOutputFile(self,value):
        self.__removeOutputFile=value
        
    @property
    def ionizationPolarity(self):
        return self.__ionizationPolarity

    @ionizationPolarity.setter
    def ionizationPolarity(self,value):
        self.__ionizationPolarity=value

        if isinstance(self.__ionizationPolarity,int):

            if self.__ionizationPolarity>0:
                self.__ionizationPolarity="+"

            elif self.__ionizationPolarity<0:
                self.__ionizationPolarity="-"
            else:
                self.__ionizationPolarity=""
        else:
            self.__ionizationPolarity="+"
            print("W.(msInsilicoFragmenter): invalid charge int type ({type(self.__ionizationPolarity)}). Assigning as positive")

  
    def __formatMZSpectrum(self):
        __msSpectrum=pd.DataFrame(columns=['mz','intensity','fragmentID','score','scoreLevel','CE'])
        __msSpectrumFragments=pd.DataFrame(columns=['fragmentID','fragmentSMILES'])
        self.__msSpectrumData=pd.DataFrame()
        if not(os.path.exists(self.msPredOutputFile)):
            print("E.(msInsilicoFragmenter): error in ms prediction")
            return
        msSpectrum=open(self.msPredOutputFile,'r').readlines()

        
        fragmentsFlag=False
        for line in msSpectrum:
            if not(re.match("^#",line[0])):
                
                if not(fragmentsFlag): line=line.replace("(","").replace(")","")
                
                line=line.split()
                
                if len(line)>0:
                    if re.match("^energy",line[0]):
                        CE=line[0].replace("energy0","10eV").replace("energy1","20eV").replace("energy2","40eV")
                    elif not(fragmentsFlag):

                        dataL=int((len(line)-2)/2)
                        for idx in range(dataL):
                            specSign=line[0:2]+[line[2+idx]]+[line[dataL+2+idx]]
                            sType='L'
                            if idx==0: sType='H'
                            specSign = [float(p) for p in specSign]+[sType,CE]
                            __msSpectrum.loc[len(__msSpectrum)]=specSign
                    else:
                        __msSpectrumFragments.loc[len(__msSpectrumFragments)]=[int(line[0]),line[2]]


                else:
                    fragmentsFlag=True

        if self.removeOutputFile: os.remove(self.msPredOutputFile)
        self.__msSpectrumData=__msSpectrum.astype({'fragmentID': 'int'}).merge(__msSpectrumFragments,on="fragmentID").drop(['fragmentID'],axis=1).sort_values('CE').reset_index(drop=True)
    
    
    def annotateSpectrum(self,smiles,ionSpecies, formula, extDB="",mz=""):

        if ionSpecies in self.__ionSpecies.keys():
            __ionSpecies=self.__ionSpecies[ionSpecies]
        else:
            return

        if ( (self.__smiles!=smiles) | (self.__metFragIonSpecies!=__ionSpecies) | (not(self.__msSpectrum4Annotation.empty))):
            self.__smiles=smiles
            self.__metFragIonSpecies=__ionSpecies

            __suffix=datetime.now().strftime('%y%m%d%H%M%S%f')
            self.__outputNameAnnot=f"msSpectrumAnnotation_{__suffix}"
            __ms=self.msSpectrum4Annotation.copy()
            __ms.intensity=__ms.intensity/__ms.intensity.max()*1000
            __ms.to_csv(f"spectrum_{__suffix}",index=False,header=False,sep=" ")
            self.__cmdArguments=f"'{self.__smiles}' -f '{formula}' -a {self.__metFragIonSpecies} -s 'spectrum_{__suffix}' -o '{self.__outputNameAnnot}' -b '{extDB}' -m {mz}"
            self.metfragAnnotateMSSpectrum(self.__cmdArguments,oFile=self.__outputNameAnnot,outpath=f"{self.__outputPath}")
            try:
                self.__annotationData=pd.read_csv(f"{self.__outputNameAnnot}.tsv",sep="\t")
            except:
                None
                
            os.remove(f"{self.__outputNameAnnot}.tsv")
            os.remove(f"spectrum_{__suffix}")
            self.__formatSpectrumAnnotations()

            


    def __formatSpectrumAnnotations(self): 
        if self.__annotationData.empty: return pd.DataFrame()

        __annotationData=self.__annotationData.copy()
        __annotationData=__annotationData.dropna(subset=["SmilesOfExplPeaks"])
       
        self.__annotationData=pd.DataFrame()
        self.__fragmentsProbs=pd.DataFrame()
        
        for candidate in __annotationData.iterrows():
            __annotationDF=pd.DataFrame({'mz':candidate[1].SmilesOfExplPeaks.split(";")})['mz'].str.split(":",expand=True).rename(columns={0:'mz',1:'smilesF'})
            tmpDF=__annotationDF.copy()
            tmpDF.insert(0,'Score',candidate[1].Score)
            tmpDF.insert(0,'inchikey',candidate[1].InChIKey)
            tmpDF.insert(0,'IDKey',candidate[1].Identifier)
            __annotationDF=pd.DataFrame({'mz':candidate[1].FormulasOfExplPeaks.split(";")})['mz'].str.split(":",expand=True)[[1]].rename(columns={1:'formula'})
            tmpDF=tmpDF.join(__annotationDF)
            tmpDF['fragmentScore']=[float(value) for value in str(candidate[1].FragmenterScore_Values).split(";")]
            self.__annotationData=pd.concat([self.__annotationData,tmpDF],sort=False,ignore_index=True)
            if "AutomatedPeakFingerprintAnnotationScore_Probtypes" in candidate[1].index:
                tmpDF=candidate[1].AutomatedPeakFingerprintAnnotationScore_Probtypes
                if str(tmpDF)!="nan":
                    tmpDF=pd.DataFrame({'p_':candidate[1].AutomatedPeakFingerprintAnnotationScore_Probtypes.split(";")})['p_'].str.split(":",expand=True)[[2,1,0]].rename(columns={1:'p',2:'mz',0:'type'})
                    tmpDF.insert(0,'IDKey',candidate[1].Identifier)
                    self.__fragmentsProbs=pd.concat([self.__fragmentsProbs,tmpDF],sort=False,ignore_index=True)
        return 
        




    def resetFragmentsData(self):
        self.__msSpectrumData=pd.DataFrame()
        self.__annotationData=pd.DataFrame()
        self.__fragmentsProbs=pd.DataFrame()


     

