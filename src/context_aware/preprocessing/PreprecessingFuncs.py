import numpy as np

from .Helpers import FindLastTransmissionIdx, DiscretizedTraffic
from ..config import DatasetConfig

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
    smoothFs = 1 / dataUnit.Ts
    smoothOrder = datasetConfig.smooth_order

    lenDataset = dataUnit.dataLength
    transmissionFlags = dataUnit.getTransmissionFlags()
    timestamps = dataUnit.getTimestamps()
    contextDataDpDr = dataUnit.getContextData()
    contextDataNoSmooth, contextData = _PrePrecessing(contextDataDpDr, transmissionFlags, timestamps, smoothFc, smoothFs, smoothOrder)

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

def _PrePrecessing(contextData, transmissionFlags, timestamps, smoothFc, smoothFs, smoothOrder):
    contextDataNoSmooth = interpolateContextData(transmissionFlags, contextData, timestamps)
    contextData_ = smoothDataByFiltfilt(contextDataNoSmooth, smoothFc, smoothFs, smoothOrder)
    contextDataSmoothed = normalizeColumns(contextData_)
    return contextDataNoSmooth, contextDataSmoothed




