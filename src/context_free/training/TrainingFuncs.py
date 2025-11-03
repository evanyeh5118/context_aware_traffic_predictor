import torch
import torch.nn as nn
import torch.optim as optim

from .Helpers import prepareTraining, trainModelHelper
from ..config import TrainingConfig

def trainModel(model, trainData, testData, trainingConfig: TrainingConfig, verbose=False):
    model, criterion, optimizer, train_loader, test_loader= prepareTraining(
        model, trainData, testData, trainingConfig, verbose=verbose)

    best_model, avg_train_loss_history, avg_test_loss_history = trainModelHelper(
         model, criterion, optimizer, train_loader, test_loader, trainingConfig, verbose=verbose)
    return best_model, avg_train_loss_history, avg_test_loss_history


