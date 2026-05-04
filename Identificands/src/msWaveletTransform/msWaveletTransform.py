# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import os
import io
import subprocess
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.environ['IDENTIFICANDS_BASEPATH'],'External/WaveletTransform/libreriaWaveletsAAren/'))
from waveletsCurr import WaveletAnalysis
from waveletsCurr import Ricker


class msWaveletTransform(object):

    def __init__(self):
        self.__WT=None
        self.__signal=np.empty((0,0))
        self.__dt=1
        self.__nsigma=2
        self.__powerTheresholdFraction=0.9
        

    @property
    def dt(self):
        return self.__dt

    @dt.setter
    def dt(self,value):
        self.__dt=value
        self.getWT()


    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self,msSpectrum):
        self.__signal=np.array(msSpectrum.intensity)

    @property
    def nsigma(self):
        return self.__nsigma

    @nsigma.setter
    def nsigma(self,value):
        self.__nsigma=value

    @property
    def powerTheresholdFraction(self):
        return self.__powerTheresholdFraction

    @powerTheresholdFraction.setter
    def powerTheresholdFraction(self,value):
        self.__powerTheresholdFraction=value

    
    @property
    def scales(self):
        if not(self.__WT is None ):
            return self.__WT.scales

    @property
    def time(self):
        if not(self.__WT is None ):
            return self.__WT.time

    @property
    def power(self):
        if not(self.__WT is None ):
            return self.__WT.wavelet_power


    def getWT(self):
        if self.__signal.shape[0]!=0:
            self.__WT = WaveletAnalysis(self.__signal, dt=self.__dt)  
   

    def getProbablePatternsLocations(self,msSpectrum,isoSignalNum=5):
        self.signal=msSpectrum
        self.getWT()
        
        zeroScalePower=self.power
        zeroScalePower=zeroScalePower[0,:]/np.max(zeroScalePower[0,:])
        sortedZeroScalePower=np.sort(zeroScalePower)
        sortedZeroScalePower=sortedZeroScalePower[0:int(len(sortedZeroScalePower)*self.__powerTheresholdFraction)] 
        powerThreshold=np.average(sortedZeroScalePower)+self.nsigma*np.std(sortedZeroScalePower) 
        msSpectrumAndWTzeroScalePower=msSpectrum.copy()
        msSpectrumAndWTzeroScalePower['zeroScalePower']=zeroScalePower

        msSpectrumAndWTzeroScalePower=msSpectrumAndWTzeroScalePower[msSpectrumAndWTzeroScalePower.zeroScalePower > powerThreshold]

        
        for idx,mz in enumerate(msSpectrumAndWTzeroScalePower.mz):
            if idx==0:
                preliminaryPatternSections=msSpectrum[(msSpectrum.mz>=mz) & (msSpectrum.mz <= mz+isoSignalNum)].copy()              
                preliminaryPatternSections['patternSectionIdx']=idx+1
            else:
                tmp=msSpectrum[(msSpectrum.mz >= mz) & (msSpectrum.mz <= mz+isoSignalNum)].copy()
                tmp['patternSectionIdx']=idx+1
        
                preliminaryPatternSections=pd.concat([preliminaryPatternSections,tmp],sort=False,ignore_index=True)

        return preliminaryPatternSections
