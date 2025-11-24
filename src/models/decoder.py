import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim * 2  # Match encoder's output dim
        self.n_layers = n_layers
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        # First layer: output_dim -> hidden_dim*2
        self.lstm_layers.append(nn.LSTM(output_dim, self.hidden_dim, num_layers=1))
        self.ln_layers.append(nn.LayerNorm(self.hidden_dim))
        # Project residual from output_dim to hidden_dim*2
        self.residual_projections.append(nn.Linear(output_dim, self.hidden_dim))

        # Subsequent layers: hidden_dim*2 -> hidden_dim*2
        for _ in range(n_layers - 1):
            self.lstm_layers.append(nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1))
            self.ln_layers.append(nn.LayerNorm(self.hidden_dim))
            self.residual_projections.append(nn.Identity())

        self.fc_out = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, hidden_states, cell_states):
        # input: [1, batch_size, output_dim]
        output = input
        next_hidden_states = []
        next_cell_states = []

        for i, layer in enumerate(self.lstm_layers):
            residual = output
            hidden = hidden_states[i]
            cell = cell_states[i]

            output, (next_hidden, next_cell) = layer(output, (hidden, cell))
            output = self.ln_layers[i](output)

            # Residual connection
            residual = self.residual_projections[i](residual)

            output = output + residual

            next_hidden_states.append(next_hidden)
            next_cell_states.append(next_cell)

        prediction = self.fc_out(output.squeeze(0))
        return prediction, next_hidden_states, next_cell_states