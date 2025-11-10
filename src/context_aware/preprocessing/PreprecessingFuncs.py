import numpy as np

from .Helpers import FindLastTransmissionIdx, DiscretizedTraffic
from ..config import DatasetConfig
from .filter import MultiDimExpSmoother

from .Helpers import normalizeColumns, interpolateContextData, smoothDataByFiltfilt

def PreparingDataset(dataUnit, datasetConfig: DatasetConfig, verbose=True):
    trainRatio = datasetConfig.train_ratio

    train_size = int(trainRatio*dataUnit.dataLength)
    dataUnitTrain = dataUnit[:train_size]
    dataUnitTest = dataUnit[train_size:]

    if verbose == True:
        print(f"Train size: {dataUnitTrain.dataLength}, Test size: {dataUnitTest.dataLength}")

    return (
        PreparingDatasetHelper(dataUnitTrain, datasetConfig),
        PreparingDatasetHelper(dataUnitTest, datasetConfig)
    )


def PreparingDatasetHelper(dataUnit, datasetConfig: DatasetConfig):
    lenSource = datasetConfig.len_source
    lenTarget = datasetConfig.len_target
    dataAugment = datasetConfig.data_augment
    smoothFc = datasetConfig.smooth_fc
    Ts = dataUnit.Ts
    max_val = np.full(dataUnit.contextData.shape[1], datasetConfig.max_val)
    min_val = np.full(dataUnit.contextData.shape[1], datasetConfig.min_val)

    lenDataset = dataUnit.dataLength
    transmissionFlags = dataUnit.getTransmissionFlags()
    timestamps = dataUnit.getTimestamps()
    contextDataDpDr = dataUnit.getContextData()
    contextDataNoSmooth, contextData = _PreProcessing(contextDataDpDr, transmissionFlags, timestamps, smoothFc, Ts, max_val, min_val)

    sources, targets, lastTranmittedContext, transmissionsVector, trafficStatesSource, trafficStatesTarget, sourcesNoSmooth = [], [], [], [], [], [], []
    if dataAugment == True:
        idxs = [(i, FindLastTransmissionIdx(transmissionFlags, i)) 
                for i in range(lenSource, lenDataset - lenTarget)]
    else:
        idxs = [(i * lenTarget, FindLastTransmissionIdx(transmissionFlags, i * lenTarget)) 
                for i in range(int(lenSource/lenTarget), int(np.floor(lenDataset / lenTarget)))]
        
    for i, last_transmission_idx in idxs:
        sources.append(contextData[i-lenSource:i])
        targets.append(contextData[i:i+lenTarget])
        transmissionsVector.append(transmissionFlags[i:i+lenTarget])
        trafficStatesSource.append(np.sum(transmissionFlags[i-lenSource:i]))
        trafficStatesTarget.append(np.sum(transmissionFlags[i:i+lenTarget]))
        lastTranmittedContext.append(contextData[last_transmission_idx:last_transmission_idx+1])
        sourcesNoSmooth.append(contextDataNoSmooth[i-lenSource:i])
        
    trafficClassesTarget = DiscretizedTraffic(trafficStatesTarget) #[0 ~ L]
    return (
        np.array(sources), 
        np.array(targets),
        np.array(lastTranmittedContext),
        np.array(trafficStatesSource).reshape(-1,1),
        np.array(trafficStatesTarget).reshape(-1,1),
        np.array(trafficClassesTarget).reshape(-1,1),
        np.array(transmissionsVector),
        np.array(sourcesNoSmooth)
    )

def _PreProcessing(contextData, transmissionFlags, timestamps, smoothFc, Ts, max_val, min_val):
    filter = MultiDimExpSmoother(fc=smoothFc, Ts=Ts, buffer_size=500)
    contextDataNoSmooth = interpolateContextData(transmissionFlags, contextData, timestamps)
    contextDataSmoothed = filter.filter(contextDataNoSmooth)
    #contextDataSmoothed = smoothDataByFiltfilt(contextDataNoSmooth, smoothFc, 1.0/Ts, 3)
    contextDataSmoothed = normalizeColumns(contextDataSmoothed, max_val, min_val)
    return contextDataNoSmooth, contextDataSmoothed




