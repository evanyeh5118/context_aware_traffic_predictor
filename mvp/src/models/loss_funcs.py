import torch.nn as nn

class CustomLossFunction(nn.Module):
    def __init__(self, lambda_ce=0.1):
        super(CustomLossFunction, self).__init__()
        self.lambda_ce = lambda_ce
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, 
        outputs_traffic, traffics,
        outputs_traffic_class, traffic_class):
        mse_loss = self.mse(outputs_traffic, traffics)
        ce_loss = self.cross_entropy(outputs_traffic_class, traffic_class)

        return mse_loss + self.lambda_ce*ce_loss, mse_loss
