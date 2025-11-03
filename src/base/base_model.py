from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch



class BaseModel(nn.Module, ABC):
    """Abstract base class for all traffic prediction models."""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        pass
    
    
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
