import numpy as np

from .Helpers import FindLastTransmissionIdx, DiscretizedTraffic
from ..config import MetaConfig
from .filter import MultiDimExpSmoother, ChunkSmoother

from .Helpers import normalizeColumns, interpolateContextData, smoothDataByFiltfilt


def PreparingDataset(dataUnit, metaConfig: MetaConfig, dataAugment=True):
    lenSource = metaConfig.window_length
    lenTarget = metaConfig.window_length
    smoothFc = metaConfig.smooth_fc
    Ts = metaConfig.Ts
    max_vals = metaConfig.max_vals
    min_vals = metaConfig.min_vals

    lenDataset = dataUnit.dataLength
    transmissionFlags = dataUnit.getTransmissionFlags()
    timestamps = dataUnit.getTimestamps()
    contextDataDpDr = dataUnit.getContextData()
    contextDataNoSmooth, contextData = _PreProcessing(contextDataDpDr, transmissionFlags, timestamps, smoothFc, Ts, max_vals, min_vals)

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

def _PreProcessing(contextData, transmissionFlags, timestamps, smoothFc, Ts, max_vals, min_vals):
    filter = MultiDimExpSmoother(fc=smoothFc, Ts=Ts, buffer_size=500)
    #filter = ChunkSmoother(dim=contextData.shape[1])
    contextDataNoSmooth = interpolateContextData(transmissionFlags, contextData, timestamps)
    contextDataSmoothed = filter.filter(contextDataNoSmooth)
    #contextDataSmoothed = filter.process(contextDataNoSmooth)
    #contextDataSmoothed = smoothDataByFiltfilt(contextDataNoSmooth, smoothFc, 1.0/Ts, 3)
    contextDataSmoothed = normalizeColumns(contextDataSmoothed, max_vals, min_vals)
    return contextDataNoSmooth, contextDataSmoothed




