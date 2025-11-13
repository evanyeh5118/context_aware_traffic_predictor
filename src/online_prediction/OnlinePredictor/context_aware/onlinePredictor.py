import numpy as np

from .dataProcessor import DataProcessor
from ..Helper import OnlineGainOptimizer

class ExpFilter:
    """Exponential filter for smoothing time-series data."""
    def __init__(self, alpha=0.3, initial_value=None):
        """
        Initialize the exponential filter.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). 
                   Higher values give more weight to recent observations.
            initial_value: Initial filtered value. If None, first observation will be used.
        """
        self.alpha = alpha
        self.filtered_value = initial_value
        self.is_initialized = initial_value is not None
    
    def update(self, new_value):
        """
        Update the filter with a new observation.
        
        Args:
            new_value: New observation value
            
        Returns:
            Filtered value
        """
        if not self.is_initialized:
            self.filtered_value = new_value
            self.is_initialized = True
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        
        return self.filtered_value
    
    def reset(self):
        """Reset the filter state."""
        self.filtered_value = None
        self.is_initialized = False
    
    def get_value(self):
        """Get the current filtered value."""
        return self.filtered_value

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
        self.exp_filter = ExpFilter(alpha=0.4)

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
        traffic_predicted = traffic_predicted[0]
        traffic_recieved = traffic_recieved[0]
        traffic_predicted = self.exp_filter.update(traffic_predicted)

        if self.last_predicted_traffic is not None and self.dataProcessor.is_ready():
            self.onlineGainOptimizer.update(traffic_recieved, self.last_predicted_traffic)
            self.gain = self.onlineGainOptimizer.get_gain()

        self.last_predicted_traffic = traffic_predicted
        return self.gain * traffic_predicted, traffic_recieved
