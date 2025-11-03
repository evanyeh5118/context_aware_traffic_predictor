"""
Model-specific configuration classes.
"""

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    len_source: int = 0
    len_target: int = 0
    train_ratio: float = 0.6
    data_augment: bool = True
    smooth_fc: float = 1.5
    smooth_order: int = 3
    @classmethod
    def initialize(cls, len_window: int, data_augment: bool):
        return cls(
            len_source=len_window,
            len_target=len_window,
            data_augment=data_augment
        )

@dataclass
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 0.01
    batch_size: int = 8192
    lambda_traffic_class: float = 100.0
    lambda_transmission: float = 500.0
    lambda_context: float = 100.0

@dataclass
class ModelConfig:
    input_size: int
    output_size: int
    hidden_size: int = 64
    num_layers: int = 5
    dropout_rate: float = 0.8
    dt: float = 0.01
    degree: int = 3
    len_source: int = 0
    len_target: int = 0
    num_classes: int = 0
    @classmethod
    def from_dataset(cls, datasetConfig: DatasetConfig, dataset):
        source_train, _, _, _, _, _, transmission_train, _ = dataset
        input_size = source_train.shape[2]
        output_size = transmission_train.shape[1]
        return cls(
            input_size=input_size,
            output_size=output_size,
            hidden_size=64,
            num_layers=5,
            dropout_rate=0.8,
            dt=0.01,
            degree=3,
            len_source=datasetConfig.len_source,
            len_target=datasetConfig.len_target,
            num_classes=datasetConfig.len_target + 1,
        )
