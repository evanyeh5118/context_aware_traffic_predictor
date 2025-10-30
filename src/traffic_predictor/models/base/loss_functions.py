"""
Custom loss functions for different model types.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class CustomLossFunction(nn.Module):
    """Base custom loss function."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss and return individual loss components."""
        raise NotImplementedError


class ContextFreeLoss(CustomLossFunction):
    """Loss function for context-free models."""
    
    def __init__(self, lambda_ce: float = 0.1):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute context-free loss."""
        mse_loss = self.mse(outputs["traffic"], targets["traffic"])
        ce_loss = self.cross_entropy(outputs["traffic_class"], targets["traffic_class"])
        
        total_loss = mse_loss + self.lambda_ce * ce_loss
        
        loss_components = {
            "mse_loss": mse_loss,
            "ce_loss": ce_loss,
            "total_loss": total_loss
        }
        
        return total_loss, loss_components


class ContextAssistedLoss(CustomLossFunction):
    """Loss function for context-assisted models."""
    
    def __init__(self, lambda_trans: float = 0.1, lambda_class: float = 0.1, lambda_context: float = 0.0):
        super().__init__()
        self.lambda_trans = lambda_trans
        self.lambda_class = lambda_class
        self.lambda_context = lambda_context
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.mse_context = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute context-assisted loss."""
        mse_loss = self.mse(outputs["traffic"], targets["traffic"])
        ce_loss = self.cross_entropy(outputs["traffic_class"], targets["traffic_class"])
        bce_loss = self.bce(outputs["transmission"], targets["transmission"])
        
        total_loss = mse_loss + self.lambda_class * ce_loss + self.lambda_trans * bce_loss
        
        loss_components = {
            "mse_loss": mse_loss,
            "ce_loss": ce_loss,
            "bce_loss": bce_loss,
            "total_loss": total_loss
        }
        
        # Add context loss if context is provided and lambda_context > 0
        if "context" in outputs and "context" in targets and self.lambda_context > 0:
            mse_context_loss = self.mse_context(outputs["context"], targets["context"])
            total_loss = total_loss + self.lambda_context * mse_context_loss
            loss_components["mse_context_loss"] = mse_context_loss
        
        loss_components["total_loss"] = total_loss
        return total_loss, loss_components
