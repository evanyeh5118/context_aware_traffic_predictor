"""
Model-specific configuration classes.
"""

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    len_window: int = 0
    len_source: int = 10
    len_target: int = 1
    train_ratio: float = 0.6
    data_augment: bool = True
    smooth_fc: float = 1.5
    smooth_order: int = 3
    @classmethod
    def initialize(cls, len_window: int, len_source: int, data_augment: bool):
        return cls(
            len_window=len_window,
            len_source=len_source,
            data_augment=data_augment
        )

@dataclass
class TrainingConfig:
    num_epochs: int = 10
    learning_rate: float = 0.005
    batch_size: int = 8192
    teacher_forcing_ratio: float = 0.2


@dataclass
class ModelConfig:
    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2