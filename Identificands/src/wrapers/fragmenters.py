import sys
import os
import subprocess
import numpy as np
import pandas as pd

class fragmentersCL(object):

    def __init__(self):
        self.adduct=1
        self.commandPath=os.path.join(os.environ['METFRAG_BASEPATH'],"bin/")
        self.command="metfrag-CL"
        

    def setFragmenter(self, fragmenter):
        self.commandPath=os.path.join(os.environ['METFRAG_BASEPATH'],"bin/")
        self.command="metfrag-CL"

        
    def metfragAnnotateMSSpectrum(self,args, oFile,removeSpectra='', outpath='./'):
        self.setFragmenter('metfrag')        
        cmd=self.commandPath+self.command
        cmd_args=" "+args
        #print("Executing: ",self.command+cmd_args)
        os.system(cmd+cmd_args)

        if removeSpectra!="":
            os.system("rm "+removeSpectra)
        if outpath!="./":
            if not os.path.isdir(outpath): 
                os.makedirs(outpath)
            os.system(f"mv {oFile} '{outpath}'")
        return 

    
