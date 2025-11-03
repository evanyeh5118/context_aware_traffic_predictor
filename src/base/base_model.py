from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
