"""
Model-specific configuration classes.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class DataProcessorConfig:
    dim_data: int = 0
    window_length: int = 0
    history_length: int = 100
    smooth_fc: float = 3.0
    smooth_order: int = 3
    Ts : float = 0.01
    min_vals: np.ndarray = np.array([])
    max_vals: np.ndarray = np.array([])
    @classmethod
    def initialize(cls, dim_data: int, window_length: int):
        return cls(
            dim_data=dim_data,
            window_length=window_length,
            min_vals=np.full(dim_data, -0.5),
            max_vals=np.full(dim_data, 0.5)
        )
    def display(self):
        print(f"DataProcessorConfig:")
        print(f"  dim_data: {self.dim_data}")
        print(f"  window_length: {self.window_length}")
        print(f"  history_length: {self.history_length}")
        print(f"  smooth_fc: {self.smooth_fc}")
        print(f"  smooth_order: {self.smooth_order}")
        print(f"  Ts: {self.Ts}")
        print(f"  min_vals: {self.min_vals}")
        print(f"  max_vals: {self.max_vals}")

@dataclass
class DatasetConfig:
    len_source: int = 0
    len_target: int = 0
    train_ratio: float = 0.7
    data_augment: bool = True
    smooth_fc: float = 3.0
    smooth_order: int = 3
    max_val: float = 0.5
    min_val: float = -0.5
    @classmethod
    def initialize(cls, len_window: int, data_augment: bool):
        return cls(
            len_source=len_window,
            len_target=len_window,
            data_augment=data_augment
        )

@dataclass
class TrainingConfig:
    num_epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 8192
    lambda_traffic_class: float = 100.0
    lambda_transmission: float = 500.0
    lambda_context: float = 50.0


@dataclass
class ModelConfig:
    input_size: int
    output_size: int
    hidden_size: int = 128
    num_layers: int = 3
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
            len_source=datasetConfig.len_source,
            len_target=datasetConfig.len_target,
            num_classes=datasetConfig.len_target + 1,
        )
