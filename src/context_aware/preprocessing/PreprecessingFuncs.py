import numpy as np

from .Helpers import FindLastTransmissionIdx, DiscretizedTraffic
from ..config import MetaConfig
from .filter import MultiDimExpSmoother, ChunkSmoother

from .Helpers import normalizeColumns, interpolateContextData, smoothDataByFiltfilt

class PreprocessingDataset:
    def __init__(self, metaConfig: MetaConfig):
        self.metaConfig = metaConfig

    def process(self, dataUnit, dataAugment=True, filterMode='filtfilt'):
        lenSource = self.metaConfig.window_length
        lenTarget = self.metaConfig.window_length

        lenDataset = dataUnit.dataLength
        transmissionFlags = dataUnit.getTransmissionFlags()
        timestamps = dataUnit.getTimestamps()
        contextDataDpDr = dataUnit.getContextData()
        contextDataNoSmooth = self._interpolateAndNormalize(contextDataDpDr, transmissionFlags, timestamps)
        contextData = self._filter(contextDataNoSmooth, filterMode)
        (
            sources, targets, lastTranmittedContext, 
            transmissionsVector, trafficStatesSource, 
            trafficStatesTarget, sourcesNoSmooth
        ) = (
            [], [], [], [], [], [], []
        )
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

    def _interpolateAndNormalize(self, contextDataDpDr, transmissionFlags, timestamps):
        contextDataInterpolated = interpolateContextData(transmissionFlags, contextDataDpDr, timestamps)
        contextDataNorm = normalizeColumns(
            contextDataInterpolated, self.metaConfig.max_vals, self.metaConfig.min_vals)
        return contextDataNorm

    def _filter(self, contextData, filterMode):
        if filterMode == 'filtfilt':
            contextDataSmoothed = smoothDataByFiltfilt(
                contextData, self.metaConfig.smooth_fc, 1.0/self.metaConfig.Ts, 3
            )
        elif filterMode == 'exp':
            filter = MultiDimExpSmoother(
                fc=self.metaConfig.smooth_fc, Ts=self.metaConfig.Ts, buffer_size=500
            )
            contextDataSmoothed = filter.filter(contextData)
        elif filterMode == 'chunk':
            print(self.metaConfig.dim_data)
            filter = ChunkSmoother(dim=self.metaConfig.dim_data)
            contextDataSmoothed = filter.process(contextData)

        return contextDataSmoothed




