import numpy as np

from ..config import MetaConfig


class PreprocessingDataset:
    def __init__(self, metaConfig: MetaConfig):
        self.metaConfig = metaConfig    
        self.lenWindow = self.metaConfig.len_window
        self.lenSource = self.metaConfig.len_source
        self.lenTarget = self.metaConfig.len_target

    def process(self, dataUnit, dataAugment=True):
        # dataset is a tuple of (dataSource, dataTarget)
        dataset = self._prepareDatasetHelper(dataUnit)
        if dataAugment == True:
            datasetAugmented = dataset
            for _ in range(self.lenSource - 1):
                dataUnit = dataUnit[1:]
                datasetNew = self._prepareDatasetHelper(dataUnit)
                datasetAugmented = self._concateDataset(datasetAugmented, datasetNew)
            dataset = datasetAugmented
        return dataset

    def _prepareDatasetHelper(self, dataUnit):      
        transmitionFlags = dataUnit.getTransmissionFlags()
        windowTraffic = self._convertToTrafficState(transmitionFlags)
        dataSource = []
        dataTarget = []
        for i in range(self.lenSource, windowTraffic.shape[0] - self.lenTarget):
            row_source = windowTraffic[i - self.lenSource:i]
            row_target = windowTraffic[i:i + self.lenTarget]
            dataSource.append(row_source)
            dataTarget.append(row_target)
        
        return np.expand_dims(np.array(dataSource), -1), np.expand_dims(np.array(dataTarget), -1)

    def _concateDataset(self, datasetA, datasetB):
        if len(datasetA) != len(datasetB):
            raise ValueError("Datasets must have the same length")
        
        datasetC = tuple(np.concatenate((a, b), axis=0) for a, b in zip(datasetA, datasetB))
        return datasetC

    def _convertToTrafficState(self, transmitionFlags):
        N_slot = int(np.floor((transmitionFlags.shape[0]) / self.lenWindow))
        windowTraffic = []
        for i in range(N_slot - 1):
            traffic_state = np.sum(transmitionFlags[i * self.lenWindow:(i + 1) * self.lenWindow])
            windowTraffic.append(traffic_state)
        return np.array(windowTraffic)


