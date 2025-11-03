import torch.optim as optim

from .Helpers import trainModelHelper, prepareTraining
from ..config import TrainingConfig

def trainModel(
    model, 
    trainData, 
    testData, 
    trainingConfig: TrainingConfig, 
    verbose=False,
    model_folder: str = None
):
    criterion, optimizer, train_loader, test_loader = prepareTraining(
        model, trainData, testData, trainingConfig, verbose=verbose)

    model,avg_train_loss_history, avg_test_loss_history = trainModelHelper(
        model, criterion, optimizer, train_loader, test_loader, 
        trainingConfig, verbose=verbose)
    return model, avg_train_loss_history, avg_test_loss_history

