"""
Online traffic predictor using a sliding window approach.
Deploys trained model for real-time inference on streaming context data.
"""

import numpy as np
import time
import json
import sys
import os
import torch
from collections import deque
from typing import Optional, Tuple, Dict, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.context_aware.config import ModelConfig, DatasetConfig
from src.context_aware.models import createModel
from src.context_aware.preprocessing.Helpers import (
    interpolateContextData,
    smoothDataByFiltfilt,
    normalizeColumns,
    FindLastTransmissionIdx
)


class RelayPredictor:
    """
    Online traffic predictor that maintains a sliding window buffer
    and performs real-time inference on streaming context data.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        window_len: int = 20,
        direction: str = "forward",
        input_size: Optional[int] = None,
        smooth_fc: float = 1.5,
        smooth_order: int = 3,
        Ts: float = 0.01,  # Sampling period
        verbose: bool = True
    ):
        """
        Initialize the relay predictor.
        
        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            config_path: Path to the dataset configuration JSON file
            window_len: Length of the sliding window for inference
            direction: Direction of traffic ("forward" or "backward")
            input_size: Number of features in context data (default: 12, or auto-detected)
            smooth_fc: Cutoff frequency for smoothing filter
            smooth_order: Order of the smoothing filter
            Ts: Sampling period (time between data points)
            verbose: Whether to print debug information
        """
        self.window_len = window_len
        self.direction = direction
        self.smooth_fc = smooth_fc
        self.smooth_order = smooth_order
        self.Ts = Ts
        self.smooth_fs = 1.0 / Ts
        self.verbose = verbose
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize data buffers
        self.context_buffer = deque(maxlen=window_len * 2)  # Store more than window_len for processing
        self.transmission_buffer = deque(maxlen=window_len * 2)
        self.timestamp_buffer = deque(maxlen=window_len * 2)
        
        # Preprocessing state
        self.context_min = None
        self.context_max = None
        
        # Performance tracking
        self.inference_times = []
        self.predictions = []
        self.num_inferences = 0
        
        # Set input_size (default to 12 if not specified)
        self.input_size = input_size if input_size is not None else 12
        
        # Model will be loaded lazily when we have data or explicitly
        self.model = None
        self.device = None
        self.model_path = model_path
        
        if self.verbose:
            print(f"RelayPredictor initialized with window_len={window_len}")
            print(f"Input size: {self.input_size} (will be auto-detected from first data if needed)")
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load the trained model from checkpoint."""
        if model_path is None:
            model_path = self.model_path
        
        if self.model is not None:
            return  # Already loaded
        
        # Create model config with current dimensions
        model_config = ModelConfig(
            input_size=self.input_size,
            output_size=self.window_len,
            hidden_size=64,
            num_layers=5,
            dropout_rate=0.8,
            dt=self.Ts,
            degree=3,
            len_source=self.window_len,
            len_target=self.window_len,
            num_classes=self.window_len + 1
        )
        
        # Create and load model
        self.model, self.device = createModel(model_config)
        
        try:
            self.model.load_checkpoint(model_path)
        except Exception as e:
            # Try alternative loading method if weights_only fails (older PyTorch versions)
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model from {model_path}: {e}, {e2}")
        
        self.model.eval()
        
        if self.verbose:
            print(f"Model loaded successfully from {model_path}")
            print(f"Device: {self.device}")
    
    def add_data_point(
        self,
        context_data: np.ndarray,
        transmission_flag: int,
        timestamp: Optional[float] = None
    ):
        """
        Add a new data point to the buffer.
        
        Args:
            context_data: Context features array (shape: (12,) or (1, 12))
            transmission_flag: Transmission flag (0 or 1)
            timestamp: Optional timestamp for this data point
        """
        # Ensure context_data is 1D array
        if context_data.ndim > 1:
            context_data = context_data.flatten()
        
        if timestamp is None:
            timestamp = time.time()
        
        self.context_buffer.append(context_data.copy())
        self.transmission_buffer.append(transmission_flag)
        self.timestamp_buffer.append(timestamp)
    
    def _preprocess_window(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the current window to prepare for inference.
        
        Returns:
            sources: Smoothed and normalized context data (window_len, 12)
            last_trans_sources: Last transmitted context (1, 12)
            sourcesNoSmooth: Interpolated but not smoothed context (window_len, 12)
        """
        if len(self.context_buffer) < self.window_len:
            raise ValueError(f"Not enough data in buffer. Need {self.window_len}, have {len(self.context_buffer)}")
        
        # Extract the last window_len points
        context_data = np.array(list(self.context_buffer)[-self.window_len:])
        transmission_flags = np.array(list(self.transmission_buffer)[-self.window_len:])
        timestamps = np.array(list(self.timestamp_buffer)[-self.window_len:])
        
        # Ensure correct shape: (window_len, num_features)
        if context_data.ndim == 1:
            context_data = context_data.reshape(-1, 1)
        elif context_data.shape[0] != self.window_len:
            context_data = context_data[-self.window_len:]
        
        # Update input_size if needed (first time)
        if self.input_size != context_data.shape[1]:
            self.input_size = context_data.shape[1]
            if self.verbose:
                print(f"Detected input_size={self.input_size}, reloading model...")
            # Reload model with correct input_size
            self.model = None
            self._load_model()
        
        # Step 1: Interpolate context data (fills missing values)
        context_no_smooth = interpolateContextData(
            transmission_flags,
            context_data,
            timestamps
        )
        
        # Ensure 2D shape
        if context_no_smooth.ndim == 1:
            context_no_smooth = context_no_smooth.reshape(-1, 1)
        
        # Step 2: Smooth the data
        context_smoothed = smoothDataByFiltfilt(
            context_no_smooth,
            self.smooth_fc,
            self.smooth_fs,
            self.smooth_order
        )
        
        # Step 3: Normalize columns
        # Update normalization parameters incrementally
        if self.context_min is None or self.context_max is None:
            self.context_min = np.min(context_smoothed, axis=0)
            self.context_max = np.max(context_smoothed, axis=0)
        else:
            # Update min/max with new data
            new_min = np.min(context_smoothed, axis=0)
            new_max = np.max(context_smoothed, axis=0)
            self.context_min = np.minimum(self.context_min, new_min)
            self.context_max = np.maximum(self.context_max, new_max)
        
        # Normalize
        denom = self.context_max - self.context_min
        denom[denom == 0.0] = 1.0
        sources = (context_smoothed - self.context_min) / denom
        
        # Find last transmission index
        current_idx = len(transmission_flags) - 1
        last_trans_idx = FindLastTransmissionIdx(
            transmission_flags,
            current_idx
        )
        
        # Get last transmitted context (use smoothed version)
        if last_trans_idx < len(context_smoothed):
            last_trans_sources = context_smoothed[last_trans_idx:last_trans_idx+1]
        else:
            # Fallback: use the first point
            last_trans_sources = context_smoothed[0:1]
        
        # Normalize last_trans_sources using same normalization
        denom_trans = self.context_max - self.context_min
        denom_trans[denom_trans == 0.0] = 1.0
        last_trans_sources = (last_trans_sources - self.context_min) / denom_trans
        
        return sources, last_trans_sources, context_no_smooth
    
    def predict(self) -> Optional[np.ndarray]:
        """
        Run inference on the current window.
        
        Returns:
            Predicted traffic values (1D array) or None if not enough data
        """
        if len(self.context_buffer) < self.window_len:
            if self.verbose:
                print(f"Insufficient data: {len(self.context_buffer)}/{self.window_len}")
            return None
        
        try:
            # Preprocess window
            sources, last_trans_sources, sourcesNoSmooth = self._preprocess_window()
            
            # Ensure model is loaded
            if self.model is None:
                self._load_model()
            
            # Run inference
            start_time = time.time()
            prediction = self.model.inference(sources, last_trans_sources, sourcesNoSmooth)
            inference_time = time.time() - start_time
            
            # Track performance
            self.inference_times.append(inference_time)
            self.predictions.append(prediction.copy())
            self.num_inferences += 1
            
            if self.verbose and self.num_inferences % 10 == 0:
                avg_time = np.mean(self.inference_times[-10:])
                print(f"Inference #{self.num_inferences}: Avg time = {avg_time:.4f}s")
            
            return prediction
            
        except Exception as e:
            if self.verbose:
                print(f"Error during inference: {e}")
            return None
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.inference_times) == 0:
            return {
                "num_inferences": 0,
                "avg_inference_time": None,
                "min_inference_time": None,
                "max_inference_time": None,
                "std_inference_time": None
            }
        
        inference_times = np.array(self.inference_times)
        return {
            "num_inferences": self.num_inferences,
            "avg_inference_time": float(np.mean(inference_times)),
            "min_inference_time": float(np.min(inference_times)),
            "max_inference_time": float(np.max(inference_times)),
            "std_inference_time": float(np.std(inference_times)),
            "total_inference_time": float(np.sum(inference_times))
        }
    
    def reset(self):
        """Reset the predictor (clear buffers and stats)."""
        self.context_buffer.clear()
        self.transmission_buffer.clear()
        self.timestamp_buffer.clear()
        self.inference_times.clear()
        self.predictions.clear()
        self.num_inferences = 0
        self.context_min = None
        self.context_max = None
        if self.verbose:
            print("Predictor reset")


def demo_usage():
    """
    Demonstration of how to use RelayPredictor.
    This shows the basic usage pattern for online inference.
    """
    # Configuration
    model_folder = "../../data/models/context_aware"
    config_path = "../../experiments/config/conbined_flows.json"
    direction = "forward"
    window_len = 20
    
    model_path = f"{model_folder}/direction_{direction}_lenWindow_{window_len}.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first or update the path.")
        return
    
    # Initialize predictor
    predictor = RelayPredictor(
        model_path=model_path,
        config_path=config_path,
        window_len=window_len,
        direction=direction,
        verbose=True
    )
    
    # Simulate streaming data
    print("\n=== Simulating Online Inference ===")
    num_samples = 100
    
    # Generate synthetic context data (12 features)
    np.random.seed(42)
    for i in range(num_samples):
        # Simulate context data (12 features)
        context_data = np.random.randn(12)
        
        # Simulate transmission flag (1 = transmission occurred, 0 = no transmission)
        transmission_flag = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Add data point
        predictor.add_data_point(
            context_data=context_data,
            transmission_flag=transmission_flag,
            timestamp=time.time()
        )
        
        # Run inference once we have enough data
        if len(predictor.context_buffer) >= window_len:
            prediction = predictor.predict()
            if prediction is not None:
                print(f"Sample {i+1}: Prediction = {prediction[0]:.4f}")
    
    # Print performance statistics
    print("\n=== Performance Statistics ===")
    stats = predictor.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_usage()

