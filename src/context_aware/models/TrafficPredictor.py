import torch
import torch.nn as nn
import torch.optim as optim

from .D2T import DeadFeaturesToTrafficLayer
from .CAdjuster import ContextAdjuster

from .Helpers import _compute_poly_matrix, _compute_poly_matrix_regularized, _compute_feature_length
from ..config import ModelConfig
from ...base.base_model import BaseModel

def createModel(parameters: ModelConfig):
    len_source = parameters.len_source
    len_target = parameters.len_target
    num_classes = parameters.num_classes    
    input_size = parameters.input_size
    output_size = parameters.output_size
    hidden_size = parameters.hidden_size
    num_layers = parameters.num_layers
    dropout_rate = parameters.dropout_rate
    dt = parameters.dt
    degree = parameters.degree

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficPredictorContextAssisted(
        input_size, hidden_size, output_size, num_classes, len_source, len_target, 
        dt, degree, device, num_layers=num_layers, dropout_rate=dropout_rate
    )
    return model, device

class TrafficPredictorContextAssisted(BaseModel):
    def __init__(self, 
                 input_size, hidden_size, output_size, num_classes,
                 len_source, len_target, dt, degree, device, num_layers=1, dropout_rate=0.5):
        BaseModel.__init__(self)  
        #self.M = _compute_poly_matrix(len_source, len_target, dt, degree, device)
        self.M = _compute_poly_matrix_regularized(len_source, len_target, dt, degree, device, penalty=1e-4)
        self.len_dbf = _compute_feature_length(len_target+1)
        self.dbf2traffic = DeadFeaturesToTrafficLayer(
            self.len_dbf, hidden_size, output_size, num_classes, len_target, 
            num_layers=num_layers, dropout_rate=dropout_rate
        ).to(device)
        self.contextAdjuster = ContextAdjuster(
            input_size, 12, num_layers=num_layers, dropout=dropout_rate
        ).to(device)    
        self.reluOut = nn.ReLU()
        self.device = device
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize neural network parameters optimally using He initialization.
        He initialization is optimal for networks with ReLU activations.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # He initialization for weight (fan-in mode, suitable for ReLU)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Zero initialization for bias
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                # Special initialization for recurrent layers
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:  # input-hidden weights
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                    elif 'weight_hh' in name:  # hidden-hidden weights
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def _ComputeDeadbandFeatures(self, data):
        # data: (T, B, F)
        T, B, F = data.shape
        mag = torch.sqrt((data ** 2).sum(dim=2))
        data_b = data.permute(1, 0, 2)
        mag_sq_b = (data_b ** 2).sum(dim=2, keepdim=True)
        dist_sq_b = mag_sq_b + mag_sq_b.transpose(1, 2) - 2 * (data_b @ data_b.transpose(1, 2))
        dist_b = torch.sqrt(dist_sq_b.clamp_min(1e-12))  # (B,T,T)
        i_idx, j_idx = torch.tril_indices(T, T, offset=-1)
        pairwise_features_b = dist_b[:, i_idx, j_idx]
        mag_b = mag.transpose(0, 1)  # (B,T)
        features_b = torch.cat([mag_b, pairwise_features_b], dim=1)

        return features_b

    def forward(self, src, last_trans_src, srcNoSmooth):
        motion_predict = (self.M.unsqueeze(0) @ src.permute(2, 0, 1)).permute(1, 2, 0)
        motion_enhanced = self.contextAdjuster(srcNoSmooth.permute(1, 0, 2)).permute(1, 0, 2)
        motion_predict = motion_predict + motion_enhanced
        motion_predict =  torch.clamp(motion_predict, 0, 1)
        motion_feature = torch.cat([motion_predict, last_trans_src], dim=0)
        db_features = self._ComputeDeadbandFeatures(motion_feature)
        traffic_est, traffic_class_est, transmission_est = self.dbf2traffic(db_features)
        traffic_est = self.reluOut(traffic_est)
        return traffic_est, traffic_class_est, transmission_est, motion_predict


    def inference(self, src, last_trans_src, srcNoSmooth):
        self.eval()
        with torch.no_grad():
            src, last_trans_src, srcNoSmooth = (torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in (src, last_trans_src, srcNoSmooth))
            src, last_trans_src, srcNoSmooth = (x.unsqueeze(0) if x.dim() == 2 else x for x in (src, last_trans_src, srcNoSmooth))
            src, last_trans_src, srcNoSmooth = map(lambda x: x.permute(1, 0, 2), (src, last_trans_src, srcNoSmooth))
            result = self.forward(src, last_trans_src, srcNoSmooth)
        
        return result[0].cpu().detach().numpy().reshape(-1), result[3].cpu().detach().numpy()