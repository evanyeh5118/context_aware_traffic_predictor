import torch
import torch.optim as optim

from ...models.ContextAssisted.TrafficPredictorEnhanced import CustomLossFunction
from ...models.Helpers import countModelParameters
from .data import createDataLoadersEnhanced
from .model import createModel
from .utils import logger


def prepareTraining(parameters, trainData, testData, verbose=False):
    """Prepare all training components with enhanced features."""
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    num_workers = parameters.get("num_workers", 4)
    pin_memory = parameters.get("pin_memory", True)

    train_loader = createDataLoadersEnhanced(
        batch_size=batch_size, dataset=trainData, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = createDataLoadersEnhanced(
        batch_size=batch_size, dataset=testData, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    model, device = createModel(parameters)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = countModelParameters(model)

    model.to(device)

    criterion = CustomLossFunction(
        lambda_trans=parameters['lambda_transmission'],
        lambda_class=parameters['lambda_traffic_class'],
        lambda_context=parameters['lambda_context']
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=parameters.get("weight_decay", 1e-4)
    )

    scheduler = None
    if parameters.get("use_scheduler", True):
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=parameters.get("scheduler_step_size", 5),
            gamma=parameters.get("scheduler_gamma", 0.95)
        )

    scaler = None
    if parameters.get("use_mixed_precision", True) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if verbose:
        logger.info(f"Training samples: {len(train_loader)}, Validation samples: {len(val_loader)}")
        logger.info(f"Device: {device}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model: {model}")
        logger.info(f"Using mixed precision: {parameters.get('use_mixed_precision', True)}")
        logger.info(f"Gradient clipping: {parameters.get('gradient_clip', 1.0)}")
        logger.info(f"Early stop patience: {parameters.get('early_stop_patience', 10)}")

    return model, criterion, optimizer, scheduler, train_loader, val_loader, device, scaler


