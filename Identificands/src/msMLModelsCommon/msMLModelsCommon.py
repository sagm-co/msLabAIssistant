import sys,os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import special
import intervals

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# Set the device to CPU
tf.config.set_visible_devices([], 'GPU')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
#from numba import cuda
import tarfile
import joblib


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class msMLModelsCommon(metaclass=SingletonMeta):

    def __init__(self):
       
        ## 1. Isotopic pattern extraction
        self.__isoPatternMLmodelFile=os.path.join(os.environ['MLMODELS_PATH'],"isotopicPatters/IsoPatterns_ANNDL_model.vd.h5")

        ## 2. Chromatophic peak identification ***
        self.__chromPeaksClassifierMLmodelFile=os.path.join(os.environ['MLMODELS_PATH'],"XIC-peakQlty/XICPeakQlty_RF_model.vd.joblib")
        

        ## 3. Retention time prediction ***
        self.__instrumentalMethod="M67_vDIA"      
        self.__retentionTimeMLModelFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/RT_XGBRegressor_model_M67vDIA.vd.joblib")
        self.__tRPredictionTransformerColumnsFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/RT_M67vDIA_columnsTransformers.vd.joblib")
        self.__tRPredictionPCATransformerFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/RT_M67vDIA_pcaTransformer.vd.joblib")
        self.__tRPrediction_selectedDescriptorsFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/RT_molecularDescriptors.vd.tsv")
        self.__tRPrediction_calCurveFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/ci_RT_XGBRegressor_model_M67vDIA.vd.tsv")
        
        ## 4. Compound methodology amenability inference***
        self.__amenabilityInferenceMLModelFile=os.path.join(os.environ['MLMODELS_PATH'],"amenability/Amenability_RF_model.vd.joblib")
        self.__amenabilityTransformerColumnsFile=os.path.join(os.environ['MLMODELS_PATH'],"amenability/Amenability_columnsTransformers.vd.joblib")
        self.__amenabilityPCATransformerFile=os.path.join(os.environ['MLMODELS_PATH'],"amenability/Amenability_pcaTransformer.vd.joblib")
        self.__amenability_selectedDescriptorsFile=os.path.join(os.environ['MLMODELS_PATH'],"amenability/Amenability_molecularDescriptors.vd.tsv")


        ## 5. Compound protonation/deprotonation inference***
        self.__ionizationSpeciesPdPMLModelFile=os.path.join(os.environ['MLMODELS_PATH'],"ionizedSpecies/ionizedSpecies_RF_model.vd.joblib")
        self.__ionizationSpeciesPdPTransformerColumnsFile=os.path.join(os.environ['MLMODELS_PATH'],"ionizedSpecies/IonizedSpecies_columnsTransformers.vd.joblib")
        self.__ionizationSpeciesPdPPCATransformerFile=os.path.join(os.environ['MLMODELS_PATH'],"ionizedSpecies/IonizedSpecies_pcaTransformer.vd.joblib")
        self.__ionizationSpeciesPdP_selectedDescriptorsFile=os.path.join(os.environ['MLMODELS_PATH'],"ionizedSpecies/IonizedSpecies_molecularDescriptors.vd.tsv")
        

        ## 6. Compound ionized species inference
        self.__ionizationSpeciesMLModelFile=os.path.join(os.environ['MLMODELS_PATH'],"Others/IonizedSpecies_RF_model.v6.joblib")
        self.__ionizationSpeciesTransformerColumnsFile=os.path.join(os.environ['MLMODELS_PATH'],"Others/IonizedSpecies_RF_columnsTransformers.v6.joblib")
        self.__ionizationSpeciesPCATransformerFile=os.path.join(os.environ['MLMODELS_PATH'],"Others/IonizedSpecies_RF_pcaTransformers.v6.joblib")
        self.__ionizationSpeciesMLModel_probThresFile=os.path.join(os.environ['MLMODELS_PATH'],"Others/IonizedSpecies_RF_model_pThresholds.v6.tsv")
        self.__ionizationSpecies_selectedDescriptorsFile=os.path.join(os.environ['MLMODELS_PATH'],"Others/IonizedSpecies_RF_model_selectedDescriptors.v6.tsv")     
        
        #Models instances
        
        ## 1. Isotopic pattern extraction
        self.__isoPatternMLModel=None

        ## 2. Chromatophic peak identification
        self.__chromPeaksClassifierMLmodel=None

        ## 3. Retention time prediction
        self.__retentionTimeMLModel=None
        self.__tRPredictionTransformerColumns=None
        self.__tRPredictionPCATransformer=None
        self.__tRPrediction_selectedDescriptors=None
        self.__tRPrediction_calCurve=None

        ## 4. Compound methodology amenability inference
        self.__amenabilityInferenceMLModel=None
        self.__amenabilityTransformerColumns=None
        self.__amenabilityPCATransformer=None
        self.__amenabilityMLModel_probThres=None
        self.__amenability_selectedDescriptors=None
        
        ## 5. Compound protonation/deprotonation inference
        self.__ionizationSpeciesPdPMLModel=None
        self.__ionizationSpeciesPdPTransformerColumns=None
        self.__ionizationSpeciesPdPPCATransformer=None
        self.__ionizationSpeciesPdPMLModel_probThres=pd.DataFrame({"ionSpecies":["+H","-H"],"probThres":[0.5,0.5]})
        self.__ionizationSpeciesPdP_selectedDescriptors=None

        ## 6. Compound ionizes species inference
        self.__ionizationSpeciesMLModel=None
        self.__ionizationSpeciesTransformerColumns=None
        self.__ionizationSpeciesPCATransformer=None
        self.__ionizationSpeciesMLModel_probThres=None
        self.__ionizationSpecies_selectedDescriptors=None


        
        
    @property
    def instrumentalMethod(self):
        print("Instrumental method for training ML models used")
        return self.__instrumentalMethod

    @instrumentalMethod.setter
    def instrumentalMethod(self,value):

        if value=="M67_PRMww":
            None
        else:
            self.__instrumentalMethod="M67_vDIA"
            self.__retentionTimeMLModelFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/RT_XGBRegressor_model_M67vDIA.vd.joblib")
            self.__tRPredictionTransformerColumnsFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/RT_M67vDIA_columnsTransformers.vd.joblib")
            self.__tRPredictionPCATransformerFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/RT_M67vDIA_pcaTransformer.vd.joblib")
            self.__tRPrediction_selectedDescriptorsFile=os.path.join(os.environ['MLMODELS_PATH'],"retentionTime/RT_molecularDescriptors.vd.tsv")


            
        self.__retentionTimeMLModel=None
        self.__tRPredictionTransformerColumns=None
        self.__tRPredictionPCATransformer=None
        self.__tRPrediction_selectedDescriptors=None        
    
        
    @property
    def isoPatternMLmodelFile(self):
        return self.__isoPatternModelFile

    @isoPatternMLmodelFile.setter
    def isoPatternMLmodelFile(self,value):
        self.__isoPatternModelFile=value
        self.__isoPatternMLModel=tf.keras.models.load_model(self.__isoPatternModelFile)
        os.system('clear')

    @property
    def chromPeaksClassifierMLmodelFile(self):
        return self.__chromPeaksClassifierMLmodelFile
    
        
    @chromPeaksClassifierMLmodelFile.setter
    def chromPeaksClassifierMLmodelFile(self,value):
        self.__chromPeaksClassifierMLmodelFile=value
        self.__chromPeaksClassifierMLmodel=joblib.load(self.__chromPeaksClassifierMLmodelFile)
        os.system('clear')
                                                  

    def classifyIsotopicSignal(self, signalFeatures):

        if self.__isoPatternMLModel is None:
            self.__isoPatternMLModel=tf.keras.models.load_model(self.__isoPatternMLmodelFile)
            os.system('clear')

        return self.__isoPatternMLModel.predict(signalFeatures,verbose=0)


    def classifyChromatographicSignal(self, hogChromSignal):
        if self.__chromPeaksClassifierMLmodel is None:
            self.__chromPeaksClassifierMLmodel=joblib.load(self.__chromPeaksClassifierMLmodelFile)
            os.system('clear')

        chromSignalQlty=pd.DataFrame(self.__chromPeaksClassifierMLmodel.predict_proba([hogChromSignal]),
                                columns=["p_noise","p_chromProblems","p_acceptableSignal"])
        chromSignalQlty['qltySignal']=chromSignalQlty.iloc[0,0:3].idxmax()=='p_acceptableSignal'
        return chromSignalQlty
                                                      

    def __get_presence_prob(self,prob):

        if len(prob)>1:
            probs_=list(map(lambda item: list(item[0]),prob))
            for idx in range(len(probs_)):
                if len(probs_[idx])==1: 
                    probs_[idx]=[0.0]+probs_[idx]
        else:
            probs_=list(prob[0])
            if len(probs_)==1: 
                probs_=[0.0]+probs_

        return np.array(probs_)

    def RetentionTimeProbability(self,m_z,s_tRInfered,s_meas=0.01,qc_tR=1.92361):
        # qc_tR=1.92361 is equivalent to p=0.95 (RMSE=0.98145)
        s_z=np.sqrt(s_meas**2+s_tRInfered**2)
        k=qc_tR/s_z
        den=0.5*(special.erf(k/(np.sqrt(2)))-special.erf(-k/(np.sqrt(2))))

        int_interval=intervals.closed(m_z-k*s_z,m_z+k*s_z) & intervals.closed(-qc_tR,qc_tR)
        p=0.0
        if not(int_interval.is_empty()):
            p=0.5*(special.erf((m_z+qc_tR)/(np.sqrt(2)*s_z))-special.erf((m_z-qc_tR)/(np.sqrt(2)*s_z)))/den

        return p
        

    def getRetentionTimePrediction(self, predictorFeatures,tR_exp=None):
        
        if self.__retentionTimeMLModel is None:
            self.__retentionTimeMLModel=joblib.load(self.__retentionTimeMLModelFile)
            os.system('clear')
            self.__tRPrediction_selectedDescriptors=(pd.read_csv(self.__tRPrediction_selectedDescriptorsFile,sep="\t"))['0'].to_list()
            self.__tRPredictionTransformerColumns=joblib.load(self.__tRPredictionTransformerColumnsFile)
            self.__tRPredictionPCATransformer=joblib.load(self.__tRPredictionPCATransformerFile)
            self.__tRPrediction_calCurve=pd.read_csv(self.__tRPrediction_calCurveFile,sep="\t")


        predictorFeatures_=predictorFeatures[self.__tRPrediction_selectedDescriptors].copy()
        preprocesedFeatures=self.__tRPredictionPCATransformer.transform(self.__tRPredictionTransformerColumns.transform(predictorFeatures_))
        tRInference=self.__retentionTimeMLModel.predict(preprocesedFeatures)
        uInfo=self.__tRPrediction_calCurve.iloc[abs(self.__tRPrediction_calCurve.tR_pred_mean-tRInference).idxmin()]
        p=None

        if tR_exp!=None:
            p=self.RetentionTimeProbability(tRInference[0]-tR_exp,uInfo.RMSError)

        tRInference=pd.DataFrame({'tR':[tRInference[0]],'U_tR':[round(uInfo['U.95'],2)],'p_tR':[p]})
        return tRInference
    

    def getAmenabilityInference(self, predictorFeatures):    
        
        if self.__amenabilityInferenceMLModel is None:
            self.__amenabilityInferenceMLModel=joblib.load(self.__amenabilityInferenceMLModelFile)
            os.system('clear')
            self.__amenability_selectedDescriptors=(pd.read_csv(self.__amenability_selectedDescriptorsFile))['0'].to_list()
            self.__amenabilityTransformerColumns=joblib.load(self.__amenabilityTransformerColumnsFile)
            self.__amenabilityPCATransformer=joblib.load(self.__amenabilityPCATransformerFile)

        predictorFeatures_=predictorFeatures[self.__amenability_selectedDescriptors].copy()
        preprocesedFeatures=self.__amenabilityPCATransformer.transform(self.__amenabilityTransformerColumns.transform(predictorFeatures_))
        amenabilityProbs=self.__amenabilityInferenceMLModel.predict_proba(preprocesedFeatures)
        amenabilityProbs=self.__get_presence_prob(amenabilityProbs)
        pred_truenessVal=[amenabilityProbs[1]>0.5]
        amenabilityInference=pd.DataFrame({'probThres':[0.5],'p':[amenabilityProbs[1]],'methodAmenable':pred_truenessVal})
        return amenabilityInference

    
    def getIonizationSpeciesPdPInference(self, predictorFeatures):
        
        if self.__ionizationSpeciesPdPMLModel is None:
            self.__ionizationSpeciesPdPMLModel=joblib.load(self.__ionizationSpeciesPdPMLModelFile)
            os.system('clear')
            self.__ionizationSpeciesPdP_selectedDescriptors=(pd.read_csv(self.__ionizationSpeciesPdP_selectedDescriptorsFile))['0'].to_list()
            self.__ionizationSpeciesPdPTransformerColumns=joblib.load(self.__ionizationSpeciesPdPTransformerColumnsFile)
            self.__ionizationSpeciesPdPPCATransformer=joblib.load(self.__ionizationSpeciesPdPPCATransformerFile)


        predictorFeatures_=predictorFeatures[self.__ionizationSpeciesPdP_selectedDescriptors].copy()
        preprocesedFeatures=self.__ionizationSpeciesPdPPCATransformer.transform(self.__ionizationSpeciesPdPTransformerColumns.transform(predictorFeatures_))
        ionSpeciesProbs=self.__ionizationSpeciesPdPMLModel.predict_proba(preprocesedFeatures)
        ionSpeciesProbs=self.__get_presence_prob(ionSpeciesProbs)
        pred_truenessVal=list(map(lambda t:t[1]>=0.5,enumerate(ionSpeciesProbs[:,1])))
        ionSpeciesInference=pd.concat([self.__ionizationSpeciesPdPMLModel_probThres[['ionSpecies','probThres']],

                                       pd.DataFrame({'p':ionSpeciesProbs[:,1],'ionicProduct':pred_truenessVal})],axis=1)


        return ionSpeciesInference



    def getIonizationSpeciesInference(self, predictorFeatures):
        
        if self.__ionizationSpeciesMLModel is None:
            self.__ionizationSpeciesMLModel=joblib.load(self.__ionizationSpeciesMLModelFile)
            os.system('clear')
            self.__ionizationSpecies_selectedDescriptors=(pd.read_csv(self.__ionizationSpecies_selectedDescriptorsFile))['Descriptor'].to_list()
            self.__ionizationSpeciesTransformerColumns=joblib.load(self.__ionizationSpeciesTransformerColumnsFile)
            self.__ionizationSpeciesPCATransformer=joblib.load(self.__ionizationSpeciesPCATransformerFile)
            self.__ionizationSpeciesMLModel_probThres=pd.read_csv(self.__ionizationSpeciesMLModel_probThresFile,sep="\t")

        predictorFeatures_=predictorFeatures[self.__ionizationSpecies_selectedDescriptors].copy()
        preprocesedFeatures=self.__ionizationSpeciesPCATransformer.transform(self.__ionizationSpeciesTransformerColumns.transform(predictorFeatures_))
        ionSpeciesProbs=self.__ionizationSpeciesMLModel.predict_proba(preprocesedFeatures)
        ionSpeciesProbs=self.__get_presence_prob(ionSpeciesProbs)
        pred_truenessVal=list(map(lambda t:t[1]>=self.__ionizationSpeciesMLModel_probThres['probThres'].to_list()[t[0]],enumerate(ionSpeciesProbs[:,1])))

        
        ionSpeciesInference=pd.concat([self.__ionizationSpeciesMLModel_probThres[['ionSpecies','probThres']],
                                        pd.DataFrame({'p':ionSpeciesProbs[:,1],'ionicProduct':pred_truenessVal})],axis=1)

        ionSpeciesInference=ionSpeciesInference[(ionSpeciesInference.ionSpecies!="+H") & (ionSpeciesInference.ionSpecies!="-H") ]

        ionSpeciesInference=pd.concat([ionSpeciesInference,self.getIonizationSpeciesPdPInference(predictorFeatures)],sort=False,ignore_index=True)

        return ionSpeciesInference
    
