import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from ..base.base_model import BaseModel
from ..base.loss_functions import ContextAssistedLoss
from ...config.model_configs import ContextAssistedConfig
from .TrafficPredictorEnhanced import TrafficPredictorContextAssisted as _CoreContextAssisted
from ..helper import createDataLoaders, countModelParameters


class TrafficPredictor(BaseModel):
    """
    BaseModel-compatible wrapper around the existing Context-Assisted predictor.

    Integrates the core implementation from `TrafficPredictorEnhanced.py` with the
    project's `BaseModel` interface and `ContextAssistedLoss`.
    """

    def __init__(self, config: ContextAssistedConfig):
        super().__init__(config)

        # Derive num_classes from target horizon (consistent with existing training funcs)
        num_classes = config.len_target + 1

        # Build the core model (keeps the proven implementation intact)
        self.core: nn.Module = _CoreContextAssisted(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_classes=num_classes,
            len_source=config.len_source,
            len_target=config.len_target,
            dt=config.dt,
            degree=config.degree,
            device=self.device,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
        ).to(self.device)

        # Loss function aligned with the shared loss definitions
        self.loss_fn = ContextAssistedLoss(
            lambda_trans=config.lambda_trans,
            lambda_class=config.lambda_class,
            lambda_context=config.lambda_context,
        )

    def forward(self, src: torch.Tensor, last_trans_src: torch.Tensor, src_no_smooth: torch.Tensor):
        """
        Args:
            src: shape [src_len, batch_size, input_dim]
            last_trans_src: shape [src_len, batch_size, 1] (or compatible)
            src_no_smooth: shape [src_len, batch_size, input_dim]

        Returns:
            Dict with keys:
              - "traffic": Tensor
              - "traffic_class": Tensor
              - "transmission": Tensor
              - "context": Tensor (the predicted motion/target sequence)
        """
        out_traffic, out_traffic_class, out_trans, out_context = self.core(
            src, last_trans_src, src_no_smooth
        )

        return {
            "traffic": out_traffic,
            "traffic_class": out_traffic_class,
            "transmission": out_trans,
            "context": out_context,
        }

    def compute_loss(self, outputs: dict, targets: dict) -> torch.Tensor:
        """
        Compute the loss using ContextAssistedLoss.

        Expected targets keys:
          - "traffic": Tensor
          - "traffic_class": LongTensor (class indices)
          - "transmission": Tensor
          - "context": Tensor (optional, used if lambda_context > 0)
        """
        # Prepare outputs dict - include context if available
        outputs_dict = {
            "traffic": outputs["traffic"],
            "traffic_class": outputs["traffic_class"],
            "transmission": outputs["transmission"],
        }
        if "context" in outputs:
            outputs_dict["context"] = outputs["context"]
        
        # Prepare targets dict - include context if available
        targets_dict = {
            "traffic": targets["traffic"],
            "traffic_class": targets["traffic_class"],
            "transmission": targets["transmission"],
        }
        if "context" in targets:
            targets_dict["context"] = targets["context"]
        
        total_loss, _ = self.loss_fn(outputs_dict, targets_dict)
        return total_loss

    @staticmethod
    def get_default_config(dataset, len_source: int, len_target: int) -> ContextAssistedConfig:
        """
        Generate default configuration based on dataset and sequence lengths.
        
        Args:
            dataset: Training dataset tuple (source_train, ..., transmission_train, ...)
            len_source: Source sequence length
            len_target: Target sequence length
            
        Returns:
            ContextAssistedConfig with default parameters
        """
        (source_train, _, _, _, _, _, transmission_train, _) = dataset
        input_size = source_train.shape[2]
        output_size = transmission_train.shape[1]
        
        return ContextAssistedConfig(
            input_size=input_size,
            output_size=output_size,
            batch_size=4096 * 2,
            hidden_size=64,
            num_layers=5,
            dropout_rate=0.8,
            num_epochs=50,
            learning_rate=0.01,
            dt=0.01,
            degree=3,
            len_source=len_source,
            len_target=len_target,
            lambda_trans=500.0,
            lambda_class=100.0,
            lambda_context=100.0,
        )

    def prepare_training(self, train_data, test_data, verbose: bool = False) -> Tuple[DataLoader, DataLoader, optim.Optimizer]:
        """
        Prepare data loaders and optimizer for training.
        
        Args:
            train_data: Training dataset
            test_data: Test/validation dataset
            verbose: Whether to print setup information
            
        Returns:
            Tuple of (train_loader, test_loader, optimizer)
        """
        train_loader = createDataLoaders(
            batch_size=self.config.batch_size,
            dataset=train_data,
            shuffle=True
        )
        test_loader = createDataLoaders(
            batch_size=self.config.batch_size,
            dataset=test_data,
            shuffle=False
        )
        
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        if verbose:
            size_model = countModelParameters(self)
            print(f"Size of train loader: {len(train_loader)}, Size of test loader: {len(test_loader)}")
            print(f"Used device: {self.device}")
            print(f"Size of model: {size_model}")
            print(self)
        
        return train_loader, test_loader, optimizer

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Average training loss for the epoch
        """
        self.train()
        total_loss = 0.0
        
        for batch in train_loader:
            sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions, sources_no_smooth = (
                data.to(self.device) for data in batch
            )
            
            # Permute to [seq_len, batch_size, features]
            sources = sources.permute(1, 0, 2)
            sources_no_smooth = sources_no_smooth.permute(1, 0, 2)
            targets = targets.permute(1, 0, 2)
            last_trans_sources = last_trans_sources.permute(1, 0, 2)
            traffics_class = traffics_class.view(-1).to(torch.long)
            
            optimizer.zero_grad()
            outputs = self.forward(sources, last_trans_sources, sources_no_smooth)
            
            targets_dict = {
                "traffic": traffics,
                "traffic_class": traffics_class,
                "transmission": transmissions,
                "context": targets,
            }
            
            loss = self.compute_loss(outputs, targets_dict)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def validate(self, test_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model on test data.
        
        Args:
            test_loader: Test/validation data loader
            
        Returns:
            Tuple of (average_total_loss, loss_components_dict)
        """
        self.eval()
        total_loss = 0.0
        total_traffic_loss = 0.0
        loss_components_sum = {}
        
        with torch.no_grad():
            for batch in test_loader:
                sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions, sources_no_smooth = (
                    data.to(self.device) for data in batch
                )
                
                sources = sources.permute(1, 0, 2)
                sources_no_smooth = sources_no_smooth.permute(1, 0, 2)
                targets = targets.permute(1, 0, 2)
                last_trans_sources = last_trans_sources.permute(1, 0, 2)
                traffics_class = traffics_class.view(-1).to(torch.long)
                
                outputs = self.forward(sources, last_trans_sources, sources_no_smooth)
                
                targets_dict = {
                    "traffic": traffics,
                    "traffic_class": traffics_class,
                    "transmission": transmissions,
                    "context": targets,
                }
                
                loss, loss_components = self.loss_fn(outputs, targets_dict)
                total_loss += loss.item()
                total_traffic_loss += loss_components["mse_loss"].item()
                
                # Accumulate all loss components
                for key, value in loss_components.items():
                    if key != "total_loss":  # Skip total_loss as we track it separately
                        if key not in loss_components_sum:
                            loss_components_sum[key] = 0.0
                        loss_components_sum[key] += value.item()
        
        avg_loss = total_loss / len(test_loader)
        avg_traffic_loss = total_traffic_loss / len(test_loader)
        avg_components = {k: v / len(test_loader) for k, v in loss_components_sum.items()}
        avg_components["total_loss"] = avg_loss
        avg_components["traffic_loss"] = avg_traffic_loss
        
        return avg_loss, avg_components

    def fit(self, train_data, test_data, verbose: bool = False) -> Tuple[Dict, List[float], List[float]]:
        """
        Train the model using the provided data.
        
        Args:
            train_data: Training dataset
            test_data: Test/validation dataset
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (best_model_state_dict, train_loss_history, test_loss_history)
        """
        train_loader, test_loader, optimizer = self.prepare_training(train_data, test_data, verbose=verbose)
        
        best_metric = float('inf')
        best_weights = None
        train_loss_history = []
        test_loss_history = []
        
        for epoch in range(self.config.num_epochs):
            # Training
            avg_train_loss = self.train_epoch(train_loader, optimizer)
            
            # Validation
            avg_test_loss, loss_components = self.validate(test_loader)
            
            train_loss_history.append(avg_train_loss)
            test_loss_history.append(avg_test_loss)
            
            if verbose:
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Validation Loss: {avg_test_loss:.4f}, "
                      f"Validation Loss (Traffic): {loss_components['traffic_loss']:.4f}")
            
            # Save best model
            if avg_test_loss < best_metric:
                best_weights = self.state_dict().copy()
                best_metric = avg_test_loss
        
        return best_weights, train_loss_history, test_loss_history

    @classmethod
    def train_with_default_config(cls, len_source: int, len_target: int, 
                                   train_data, test_data, verbose: bool = False):
        """
        Convenience method to train with default configuration.
        
        Args:
            len_source: Source sequence length
            len_target: Target sequence length
            train_data: Training dataset
            test_data: Test/validation dataset
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (best_model_state_dict, train_loss_history, test_loss_history, config)
        """
        config = cls.get_default_config(train_data, len_source, len_target)
        model = cls(config)
        best_weights, train_loss_history, test_loss_history = model.fit(
            train_data, test_data, verbose=verbose
        )
        return best_weights, train_loss_history, test_loss_history, config


