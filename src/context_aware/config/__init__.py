"""
Configuration management for the traffic predictor package.
"""

from .model_configs import ModelConfig, TrainingConfig, DatasetConfig

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
]
