import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        # First layer: input_dim -> hidden_dim*2
        self.lstm_layers.append(
            nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True)
        )
        self.ln_layers.append(nn.LayerNorm(hidden_dim * 2))
        # For the first layer, we need to project the residual from input_dim to hidden_dim*2
        self.residual_projections.append(nn.Linear(input_dim, hidden_dim * 2))

        # Subsequent layers: hidden_dim*2 -> hidden_dim*2
        for _ in range(n_layers - 1):
            self.lstm_layers.append(nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_dim * 2))
            self.residual_projections.append(nn.Identity())

    def forward(self, src):
        # src: [src_len, batch_size, input_dim]
        output = src
        hidden_states = []
        cell_states = []

        for i, layer in enumerate(self.lstm_layers):
            residual = output
            output, (hidden, cell) = layer(output)
            output = self.ln_layers[i](output)

            # Residual connection
            residual = self.residual_projections[i](residual)
            output = output + residual

            hidden_states.append(self._combine_directions(hidden))
            cell_states.append(self._combine_directions(cell))

        return output, hidden_states, cell_states

    def _combine_directions(self, tensor):
        # tensor: [2, batch_size, hidden_dim]
        forward = tensor[0:1]  # [1, batch_size, hidden_dim]
        backward = tensor[1:2] # [1, batch_size, hidden_dim]
        return torch.cat((forward, backward), dim=2) # [1, batch_size, hidden_dim * 2]