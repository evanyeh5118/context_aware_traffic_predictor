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
        try:
            # Try to load with weights_only=True (more secure) and map to CPU
            checkpoint = torch.load(path, weights_only=True, map_location=torch.device('cpu'))
        except (RuntimeError, ValueError, TypeError):
            # Fall back to weights_only=False if the above fails
            # This is necessary for models saved with older PyTorch versions
            print("[WARNING] Loading checkpoint with weights_only=False due to compatibility issues")
            checkpoint = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint["model_state_dict"])
