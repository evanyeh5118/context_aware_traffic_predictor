import numpy as np

from ..config import DatasetConfig

def PreparingDataset(dataUnit, datasetConfig: DatasetConfig, verbose=True):
    trainRatio = datasetConfig.train_ratio
    lenWindow = datasetConfig.len_window
    train_size = int(trainRatio*dataUnit.dataLength)

    dataUnitTrain = dataUnit[:train_size]
    dataUnitTest = dataUnit[train_size:]

    if verbose == True:
        print(f"Train size: {dataUnitTrain.dataLength}, Test size: {dataUnitTest.dataLength}")

    dataTrain = PreparingDatasetHelper(dataUnitTrain, datasetConfig)
    dataTest = PreparingDatasetHelper(dataUnitTest, datasetConfig)
    if datasetConfig.data_augment == True:
        dataTrainAugmented = dataTrain
        dataTestAugmented = dataTest
        for i in range(lenWindow-1):
            dataUnitTrain = dataUnitTrain[1:]
            dataUnitTest = dataUnitTest[1:]
            dataTrainNew = PreparingDatasetHelper(dataUnitTrain, datasetConfig)
            dataTestNew = PreparingDatasetHelper(dataUnitTest, datasetConfig)
            dataTrainAugmented = concateDataset(dataTrainAugmented, dataTrainNew)
            dataTestAugmented = concateDataset(dataTestAugmented, dataTestNew)
        dataTrain = dataTrainAugmented
        dataTest = dataTestAugmented

    return (dataTrain, dataTest)

def PreparingDatasetHelper(dataUnit, datasetConfig: DatasetConfig):
    lenSource = datasetConfig.len_source 
    lenTarget = datasetConfig.len_target 
    transmitionFlags = dataUnit.getTransmissionFlags()
    windowTraffic = ConvertToTrafficState(transmitionFlags, datasetConfig)
    dataSource = []
    dataTarget = []
    
    for i in range(lenSource, windowTraffic.shape[0]-lenTarget):
        row_source = windowTraffic[i-lenSource:i]
        row_target = windowTraffic[i:i+lenTarget]
        dataSource.append(row_source)
        dataTarget.append(row_target)
    return np.expand_dims(np.array(dataSource), -1), np.expand_dims(np.array(dataTarget),-1)

def concateDataset(datasetA, datasetB):
    if len(datasetA) != len(datasetB):
        raise ValueError("Datasets must have the same length")
    
    datasetC = tuple(np.concatenate((a, b), axis=0) for a, b in zip(datasetA, datasetB))
    return datasetC


def ConvertToTrafficState(transmitionFlags, datasetConfig: DatasetConfig):
    lenWindow = datasetConfig.len_window

    N_slot = int(np.floor((transmitionFlags.shape[0])/lenWindow))
    windowTraffic = []
    for i in range(N_slot-1):
        traffic_state = np.sum(transmitionFlags[i*lenWindow:(i+1)*lenWindow])
        windowTraffic.append(traffic_state)
    return np.array(windowTraffic)


