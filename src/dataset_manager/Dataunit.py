from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .DeadbandReduction import DataReductionForDataUnit
from .Helper import interpolationData, resampleData, smoothDataByFiltfilt

class DataUnit:
    def __init__(self):
        self.name = []
        self.Ts = []
        self.timestamps = []
        self.contextData = []
        self.contextDataDpDr = []
        self.contextDataPorcessed = []
        self.transmitionFlags = []
        self.dimFeatures = []
        self.dataLength = []
        self.compressionRate = []

    def __getitem__(self, key):
        dataUnitCopy = DataUnit()

        dataUnitCopy.contextData = self.contextData[key]
        dataUnitCopy.contextDataDpDr = self.contextDataDpDr[key]
        dataUnitCopy.contextDataPorcessed = self.contextDataPorcessed[key]
        dataUnitCopy.transmitionFlags = self.transmitionFlags[key]

        dataUnitCopy.name = self.name
        dataUnitCopy.Ts = self.Ts
        dataUnitCopy.dataLength = dataUnitCopy.contextData.shape[0]
        dataUnitCopy.compressionRate = np.sum(dataUnitCopy.transmitionFlags) / len(dataUnitCopy.transmitionFlags)
        dataUnitCopy.dimFeatures = self.dimFeatures
        dataUnitCopy.timestamps = self.timestamps[key] - self.timestamps[0]
        return dataUnitCopy

    def setContextData(self, contextData):
        self.contextData = contextData
        self.dataLength = contextData.shape[0]
        self.dimFeatures = contextData.shape[1]
        
    def getContextDataProcessed(self):
        data = self.contextDataPorcessed.copy()
        min_values = data.min(axis=0)
        max_values = data.max(axis=0)
        normalizedData = (data - min_values) / (max_values - min_values)
        if normalizedData.ndim == 1:
            normalizedData  = normalizedData[..., np.newaxis]
        return normalizedData
    
    def getContextDataProcessedAndSmoothed(self, fc, order): #self.contextDataPorcessed -> #self.contextDataPorcessedSmoothed
        smoothData = smoothDataByFiltfilt(self.contextDataPorcessed, fc, 1/self.Ts, order)
        min_values = smoothData.min(axis=0)
        max_values = smoothData.max(axis=0)
        normalizedData = (smoothData - min_values) / (max_values - min_values)
        if normalizedData.ndim == 1:
            normalizedData  = normalizedData[..., np.newaxis]
        return normalizedData

    def getTransmissionFlags(self):
        return self.transmitionFlags.copy()

    def display(self):
        print(f"Name: {self.name}, Ts:{self.Ts}, Data length:{self.dataLength}, Dim of context:{self.dimFeatures}, Compression rate:{self.compressionRate}")

    def generateTrafficPattern(self, lenWindow):
        traffic_state = []
        N_slot = int(np.floor(len(self.transmitionFlags)/lenWindow))
        for i in range(N_slot):
            traffic_state.append(np.sum(self.transmitionFlags[i*lenWindow:(i+1)*lenWindow]))
        return np.array(traffic_state)
    
    def interpolateCotextAfterDpDr(self): #self.contextDataDpDr -> #self.contextDataPorcessed
        self.contextDataPorcessed = interpolationData(
            np.asarray(self.transmitionFlags).astype(int), 
            np.asarray(self.contextDataDpDr, dtype=np.float64), 
            np.asarray(self.timestamps))
        
    def applyDpDr(self, dbParameter=0.01, alpha=0.01, mode="fixed"): #self.contextData -> #self.contextDataDpDr
        contextDataDpDr, transmitionFlags = DataReductionForDataUnit(self, dbParameter=dbParameter, alpha=alpha, mode=mode)
        self.contextDataDpDr = contextDataDpDr
        self.transmitionFlags = transmitionFlags
        self.compressionRate = np.sum(self.transmitionFlags) / self.transmitionFlags.shape[0]

    def resampleContextData(self): #self.contextData -> self.contextData
        self.Ts = round(np.mean(self.timestamps[1:]-self.timestamps[0:-1]), 2)
        (_, self.contextData) = resampleData(self.timestamps, self.contextData, self.Ts)

