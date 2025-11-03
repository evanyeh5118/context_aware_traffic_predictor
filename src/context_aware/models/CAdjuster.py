import torch.nn as nn

class ContextAdjuster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super(ContextAdjuster, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.Sequential()

        # First Layer
        self.layers.add_module("Linear_1", nn.Linear(input_dim, hidden_dim))
        self.layers.add_module("BatchNorm_1", nn.BatchNorm1d(hidden_dim))
        self.layers.add_module("ReLU_1", nn.ReLU())
        self.layers.add_module("Dropout_1", nn.Dropout(dropout))

        # Hidden Layers
        for i in range(2, num_layers + 1):
            self.layers.add_module(f"Linear_{i}", nn.Linear(hidden_dim, hidden_dim))
            self.layers.add_module(f"BatchNorm_{i}", nn.BatchNorm1d(hidden_dim))
            self.layers.add_module(f"ReLU_{i}", nn.ReLU())
            self.layers.add_module(f"Dropout_{i}", nn.Dropout(dropout))

        # Output Layer (same shape as input_dim but different src_len)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input to (batch_size * src_len, input_dim)
        batch_size, src_len, input_dim = x.size()
        x = x.view(batch_size * src_len, input_dim)
        x = self.layers(x)
        x = self.output_layer(x)
        x = x.view(batch_size, src_len, input_dim)
        x = self.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)