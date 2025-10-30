"""
Base model class for all traffic prediction models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ...config.model_configs import ModelConfig


class BaseModel(nn.Module, ABC):
    """Abstract base class for all traffic prediction models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for the model."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.__class__.__name__,
            "config": self.config.dict(),
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.dict(),
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
