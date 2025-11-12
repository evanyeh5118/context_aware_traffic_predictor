from .dataProcessor import DataProcessor
from ..Helper import OnlineGainOptimizer

class OnlinePredictor:
    def __init__(self, model, metaConfig):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
        self.model.to(self.device)
        self.dataProcessor = DataProcessor(metaConfig)
        self.onlineGainOptimizer = OnlineGainOptimizer(
            gain_init=1.0, lr=0.001, gain_min=0.0, gain_max=1.5)
        
        self.gain = 1.0
        self.last_predicted_traffic = None

    def is_ready(self):
        return self.dataProcessor.is_ready()

    def receive_signal(self):
         self.dataProcessor.receive_signal()

    def predict(self): 
        traffic_historical = self.dataProcessor.get_historical_traffic()
        traffic_recieved = traffic_historical[-1]
        traffic_predicted = self.model.inference(traffic_historical)

        if self.last_predicted_traffic is not None and self.dataProcessor.is_ready():
            self.onlineGainOptimizer.update(traffic_recieved, self.last_predicted_traffic)
            self.gain = self.onlineGainOptimizer.get_gain()

        self.last_predicted_traffic = traffic_predicted
        return self.gain * traffic_predicted, traffic_recieved
