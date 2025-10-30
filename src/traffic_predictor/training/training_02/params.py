import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def getDefaultModelParams(len_source, len_target, dataset, batch_size: int = 1024,
                         learning_rate: float = 1e-3, dropout_rate: float = 0.2):
    """Get default model parameters with more reasonable defaults."""
    (source_train, _, _, _, _, _, transmission_train, _) = dataset
    input_size = source_train.shape[2]
    output_size = transmission_train.shape[1]
    parameters = {
        "input_size": input_size,
        "output_size": output_size,
        "batch_size": batch_size,
        "hidden_size": 64,
        "num_layers": 5,
        "dropout_rate": dropout_rate,
        "num_epochs": 50,
        "learning_rate": learning_rate,
        "dt": 0.01,
        "degree": 3,
        "len_source": len_source,
        "len_target": len_target,
        "num_classes": len_target + 1,
        "train_ratio": 0.6,
        "lambda_traffic_class": 100,
        "lambda_transmission": 500,
        "lambda_context": 100.0,
        "use_mixed_precision": True,
        "gradient_clip": 1.0,
        "early_stop_patience": 10,
        "use_scheduler": True,
        "scheduler_gamma": 0.95,
        "scheduler_step_size": 5,
        "weight_decay": 1e-4,
        "num_workers": 4,
        "pin_memory": True,
        "seed": 42,
        "checkpoint_dir": "Results/models/checkpoints",
        "save_checkpoints": True
    }
    return parameters


