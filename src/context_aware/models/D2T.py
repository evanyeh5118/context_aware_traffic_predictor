import torch
import torch.nn as nn

class DeadFeaturesToTrafficLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes, target_len_seq, num_layers=1, dropout_rate=0.5):
        super(DeadFeaturesToTrafficLayer, self).__init__()
        self.num_classes = num_classes
        self.target_len_seq = target_len_seq
        # Input layer with batch normalization and dropout
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Hidden layers with residual connections, batch normalization, and dropout
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Batch normalization
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
        # Output layer
        self.trans2transmission_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.trans2traffic_layer = nn.Linear(hidden_size + self.target_len_seq, 1)
        self.trans2trafficClass_layer = nn.Linear(hidden_size + self.target_len_seq, self.num_classes)
        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Initial layer
        x = self.input_layer(x)
        residual = x  # Save initial input as residual for skip connections
        
        # Process through hidden layers with skip connections
        for layer in self.hidden_layers:
            out = layer(residual)
            residual = out + residual  # Add skip connection if dimensions match

        # Output layer
        x_transmission = self.sigmoid(self.trans2transmission_layer(residual))
        cat_features = torch.cat((x_transmission, residual), -1)
        x_traffic = self.trans2traffic_layer(cat_features)
        x_traffic_class = self.trans2trafficClass_layer(cat_features)
        
        return x_traffic, x_traffic_class, x_transmission

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)