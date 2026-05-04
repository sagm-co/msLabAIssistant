import sys,os
import pandas as pd
import numpy as np


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class msCommonStructureData(metaclass=SingletonMeta):
 

    def pdToDict(self,df,tupleKeyValueColumns):
        keyValue=lambda i :  (df[tupleKeyValueColumns[0]].iloc[i],df[tupleKeyValueColumns[1]].iloc[i])
        dictT=dict(map(keyValue,range(len(df))))
        return dictT
