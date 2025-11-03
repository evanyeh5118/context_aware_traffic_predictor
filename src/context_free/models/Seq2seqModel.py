import torch
import torch.nn as nn
import random

from .encoder import Encoder
from .decoder import Decoder
from ..config import ModelConfig

def createModel(modelConfig: ModelConfig): 
    inputFeatureSize = modelConfig.input_size
    outputFeatureSize = modelConfig.output_size 
    hidden_size = modelConfig.hidden_size
    num_layers = modelConfig.num_layers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(
        inputFeatureSize, outputFeatureSize, hidden_size, num_layers
    ).to(device)
    return model, device

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, n_layers) 
        self.decoder = Decoder(output_dim, hidden_dim, n_layers)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size, input_dim]
        # trg: [trg_len, batch_size, output_dim]
        _, hidden_states, cell_states = self.encoder(src)

        trg_len, batch_size, output_dim = trg.size()
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(src.device)
        #input = src[-1, :, :].unsqueeze(0)  # Start decoding from the last encoder state
        input = torch.zeros(1, batch_size, output_dim).to(src.device)
        for t in range(trg_len):
            output, hidden_states, cell_states = self.decoder(input, hidden_states, cell_states)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t].unsqueeze(0) if teacher_force else output.unsqueeze(0)

        return outputs
    

