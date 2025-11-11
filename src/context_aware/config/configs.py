"""
Model-specific configuration classes.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MetaConfig:
    dim_data: int = 0
    window_length: int = 0
    history_length: int = 100
    smooth_fc: float = 0.0
    degree: int = 0
    Ts: float = 0.0  # Sampling period in seconds
    min_vals: np.ndarray = field(default_factory=lambda: np.array([]))
    max_vals: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def initialize(cls, dim_data: int, window_length: int, 
                   history_length: int = 100,
                   smooth_fc: float = 1.5,
                   degree: int = 3,
                   Ts: float = 0.01,
                   min_vals: np.ndarray = np.full(dim_data, -0.5),
                   max_vals: np.ndarray = np.full(dim_data, 0.5)):
        return cls(
            dim_data=dim_data,
            window_length=window_length,
            history_length=history_length,
            smooth_fc=smooth_fc,
            degree=degree,
            Ts=Ts,
            min_vals=min_vals,
            max_vals=max_vals,
        )
    
    def display(self):
        """Display configuration parameters."""
        print("================================================")
        print(f"MetaConfig:")
        print(f"  dim_data: {self.dim_data}")
        print(f"  window_length: {self.window_length}")
        print(f"  history_length: {self.history_length}")
        print(f"  smooth_fc: {self.smooth_fc}")
        print(f"  degree: {self.degree}")
        print(f"  Ts: {self.Ts}")
        print(f"  min_vals: {self.min_vals}")
        print(f"  max_vals: {self.max_vals}")
        print("================================================")


@dataclass
class TrainingConfig:
    num_epochs: int = 100
    learning_rate: float = 0.005
    batch_size: int = 8192
    lambda_traffic_class: float = 250.0
    lambda_transmission: float = 5000.0
    lambda_context: float = 100.0
    
    def __post_init__(self):
        """Validate training configuration parameters."""
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"learning_rate must be in (0, 1), got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lambda_traffic_class < 0:
            raise ValueError(f"lambda_traffic_class must be non-negative")
        if self.lambda_transmission < 0:
            raise ValueError(f"lambda_transmission must be non-negative")
        if self.lambda_context < 0:
            raise ValueError(f"lambda_context must be non-negative")


@dataclass
class ModelConfig:
    input_size: int
    output_size: int
    len_source: int
    len_target: int
    num_classes: int
    hidden_size: int = 256
    num_layers: int = 4
    dropout_rate: float = 0.8
    degree: int = 3
    dt : float = 0.01

    @classmethod
    def from_meta_config(cls, metaConfig: MetaConfig):
        return cls(
            input_size=metaConfig.dim_data,
            output_size=metaConfig.window_length,
            degree = metaConfig.degree,
            dt = metaConfig.Ts,
            len_source=metaConfig.window_length,
            len_target=metaConfig.window_length,
            num_classes=metaConfig.window_length + 1,
        )
