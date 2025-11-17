import numpy as np

from .Helpers import FindLastTransmissionIdx, DiscretizedTraffic
from ..config import MetaConfig

from .Helpers import normalizeColumns, interpolateContextData, smoothDataByFiltfilt
from .filter import TikhonovSmoother

class PreprocessingDataset:
    def __init__(self, metaConfig: MetaConfig):
        self.metaConfig = metaConfig
        self.smoother = TikhonovSmoother(
            self.metaConfig.dim_data, lam=0.1, dt=self.metaConfig.Ts)

    def process(self, dataUnit, dataAugment=True, pre_filter=True):
        lenSource = self.metaConfig.window_length
        lenTarget = self.metaConfig.window_length

        lenDataset = dataUnit.dataLength
        transmissionFlags = dataUnit.getTransmissionFlags()
        timestamps = dataUnit.getTimestamps()
        contextDataDpDr = dataUnit.getContextData()
        contextDataNoSmooth = self._interpolateAndNormalize(contextDataDpDr, transmissionFlags, timestamps)
        if pre_filter == True:
            contextData = smoothDataByFiltfilt(
                contextDataNoSmooth, self.metaConfig.smooth_fc, 1.0/self.metaConfig.Ts, 3
            )
        else:
            contextData = contextDataNoSmooth
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
            source = contextData[i-lenSource:i]
            if pre_filter != True:
                source = self.smoother.smooth(source)
            sources.append(source)
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





