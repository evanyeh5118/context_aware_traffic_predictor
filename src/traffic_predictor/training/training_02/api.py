from pathlib import Path
import torch

from .params import set_seed, getDefaultModelParams
from .prepare import prepareTraining
from .loop import trainModelHelper


def trainModelByDefaultSetting(len_source, len_target, trainData, testData,
                               verbose=False, save_path=None):
    parameters = getDefaultModelParams(len_source, len_target, trainData)
    best_model, histories, training_info = trainModel(
        parameters, trainData, testData, verbose=verbose, save_path=save_path
    )
    return best_model, histories, training_info


def trainModel(parameters, trainData, testData, verbose=False, save_path=None):
    if "seed" in parameters:
        set_seed(parameters["seed"])

    model, criterion, optimizer, scheduler, train_loader, val_loader, device, scaler = prepareTraining(
        parameters, trainData, testData, verbose=verbose
    )

    checkpoint_dir = Path(parameters.get("checkpoint_dir", "Results/models/checkpoints"))
    if parameters.get("save_checkpoints", True):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model, histories, training_info = trainModelHelper(
        parameters=parameters,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose
    )

    if save_path is not None:
        torch.save(best_model.state_dict(), save_path)

    return best_model, histories, training_info


