import torch
import torch.nn as nn
import random

from .encoder import Encoder
from .decoder import Decoder
from ..config import ModelConfig

from ...base.base_model import BaseModel

def createModel(modelConfig: ModelConfig): 
    inputFeatureSize = modelConfig.input_size
    outputFeatureSize = modelConfig.output_size 
    hidden_size = modelConfig.hidden_size
    num_layers = modelConfig.num_layers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(
        inputFeatureSize, outputFeatureSize, hidden_size, num_layers, device
    ).to(device)
    return model, device

class Seq2Seq(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, device):
        BaseModel.__init__(self)
        self.encoder = Encoder(input_dim, hidden_dim, n_layers) 
        self.decoder = Decoder(output_dim, hidden_dim, n_layers)
        self.device = device

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

    def inference(self, src):
        # src: (len_source,) -> (len_source, 1, 1)
        self.eval()
        with torch.no_grad():
            src = torch.as_tensor(src, dtype=torch.float32, device=self.device)
            src = src.unsqueeze(1).unsqueeze(2).to(self.device)
            dummy_targets = torch.zeros(1, 1, 1).to(self.device)
            predictions = self.forward(src, dummy_targets, teacher_forcing_ratio=0.0)
            return predictions.detach().cpu().numpy().squeeze()

        
