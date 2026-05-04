import sys
import os
import subprocess
import numpy as np
import pandas as pd

class MsSectionsNormalization(object):

    def __init__(self):
        self.logTransformLevel=3

    def getFullyNormalizedSection(self,section):
        intensityNormMsSection=section.intensity/np.max(section.intensity)
        mzNormMsSection=section.mz-min(section.mz)
        invIntensityNormMsSection=1/np.log2(np.log2(np.log2(intensityNormMsSection*10E8)))
        normalizedSection=np.array([mzNormMsSection,intensityNormMsSection,invIntensityNormMsSection]).transpose()
        return normalizedSection

