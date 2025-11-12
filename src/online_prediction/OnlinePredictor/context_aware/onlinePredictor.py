import numpy as np

from .dataProcessor import DataProcessor
from ..Helper import OnlineGainOptimizer

class OnlinePredictor:
    def __init__(self, model, metaConfig):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
        self.model.to(self.device)
        self.dataProcessor = DataProcessor(metaConfig)
        self.Ts = metaConfig.Ts
        self.window_length = metaConfig.window_length
        self.onlineGainOptimizer = OnlineGainOptimizer(
            gain_init=1.0, lr=0.001, gain_min=0.0, gain_max=1.5)
        
        self.gain = 1.0
        self.count = 0
        self.last_predicted_traffic = None

    def receive(self, data):
         self.dataProcessor.add_data_point(data)

    def predict(self): 
        (
            context, last_trans_sources, context_no_smooth, debugs
        ) = self.dataProcessor.get_window_features()
        (flags, _) = debugs

        traffic_predicted, _ = self.model.inference(
            context, last_trans_sources, context_no_smooth
        )
        traffic_recieved = np.sum(flags).reshape(-1)

        if self.last_predicted_traffic is not None and self.dataProcessor.is_ready():
            self.onlineGainOptimizer.update(traffic_recieved, self.last_predicted_traffic)
            self.gain = self.onlineGainOptimizer.get_gain()

        self.last_predicted_traffic = traffic_predicted
        return self.gain * traffic_predicted[0], traffic_recieved[0]
