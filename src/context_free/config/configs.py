"""
Model-specific configuration classes.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MetaConfig:
    dim_data: int = 1
    len_window: int = 0
    len_source: int = 10
    len_target: int = 1
    train_ratio: float = 0.6
    data_augment: bool = True
    Ts: float = 0.01  # Sampling period in seconds

    @classmethod
    def initialize(cls, len_window=20, len_source=10, len_target=1,
                   dim_data=1, train_ratio=0.6, data_augment=True, Ts=0.01):
        return cls(
            dim_data=dim_data,
            len_window=len_window,
            len_source=len_source,
            len_target=len_target,
            train_ratio=train_ratio,
            data_augment=data_augment,
            Ts=Ts,
        )
    
    def display(self):
        """Display configuration parameters."""
        print("================================================")
        print(f"MetaConfig:")
        print(f"  dim_data: {self.dim_data}")
        print(f"  len_window: {self.len_window}")
        print(f"  len_source: {self.len_source}")
        print(f"  len_target: {self.len_target}")
        print(f"  train_ratio: {self.train_ratio}")
        print(f"  data_augment: {self.data_augment}")
        print(f"  Ts: {self.Ts}")
        print("================================================")


@dataclass
class DatasetConfig:
    len_window: int = 0
    len_source: int = 10
    len_target: int = 1
    train_ratio: float = 0.6
    data_augment: bool = True
    
    @classmethod
    def initialize(cls, len_window: int, len_source: int, data_augment: bool):
        return cls(
            len_window=len_window,
            len_source=len_source,
            data_augment=data_augment
        )
    
    @classmethod
    def from_meta_config(cls, metaConfig: MetaConfig):
        return cls(
            len_window=metaConfig.len_window,
            len_source=metaConfig.len_source,
            len_target=metaConfig.len_target,
            train_ratio=metaConfig.train_ratio,
            data_augment=metaConfig.data_augment,
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

    @classmethod
    def from_meta_config(cls, metaConfig: MetaConfig):
        return cls(
            input_size=metaConfig.dim_data,
            output_size=metaConfig.dim_data,
        )