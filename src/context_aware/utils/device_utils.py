"""
Device management utilities.
"""

import torch
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", or None)
        
    Returns:
        PyTorch device object
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_device(device: str) -> torch.device:
    """
    Set the default device for computation.
    
    Args:
        device: Device specification
        
    Returns:
        PyTorch device object
    """
    torch_device = get_device(device)
    torch.set_default_device(torch_device)
    return torch_device


def print_device_info() -> None:
    """Print information about available devices."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
