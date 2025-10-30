import torch.nn as nn

class CustomLossFunction(nn.Module):
    def __init__(self, lambda_trans=0.1, lambda_class=0.1, lambda_context=0.1):
        super(CustomLossFunction, self).__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.mse_context = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.lambda_trans = lambda_trans
        self.lambda_class = lambda_class
        self.lambda_context = lambda_context

    def forward(self, 
        outputs_traffic, traffics,
        outputs_traffic_class, traffic_class,
        outputs_transmissions, transmissions,
        outputs_context, context):
        mse_loss = self.mse(outputs_traffic, traffics)
        ce_loss = self.cross_entropy(outputs_traffic_class, traffic_class)
        bce_loss = self.bce(outputs_transmissions, transmissions)       
        mse_context_loss = self.mse_context(outputs_context, context)

        return mse_loss + self.lambda_class*ce_loss + self.lambda_trans*bce_loss + self.lambda_context*mse_context_loss, mse_loss