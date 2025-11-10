import torch.optim as optim

from .Helpers import trainModelHelper, prepareTraining
from ..config import TrainingConfig

def trainModel(
    model, 
    trainData, 
    testData, 
    trainingConfig: TrainingConfig, 
    model_path=None,
    verbose=False
):
    criterion, optimizer, train_loader, test_loader = prepareTraining(
        model, trainData, testData, trainingConfig, verbose=verbose)

    model,avg_train_loss_history, avg_test_loss_history = trainModelHelper(
        model, criterion, optimizer, train_loader, test_loader, 
        trainingConfig, verbose=verbose, model_path=model_path)
    return model, avg_train_loss_history, avg_test_loss_history

