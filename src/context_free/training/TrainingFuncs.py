import torch
import torch.nn as nn
import torch.optim as optim

from .Helpers import prepareTraining, trainModelHelper
from ..config import TrainingConfig

def trainModel(model, trainData, testData, trainingConfig: TrainingConfig, verbose=False, model_path=None):
    model, criterion, optimizer, train_loader, test_loader= prepareTraining(
        model, trainData, testData, trainingConfig, verbose=verbose
    )

    model, avg_train_loss_history, avg_test_loss_history = trainModelHelper(
         model, criterion, optimizer, train_loader, test_loader, trainingConfig, verbose=verbose, model_path=model_path
    )
    return model, avg_train_loss_history, avg_test_loss_history


