import time
from collections import deque
from typing import Optional, Tuple
from src.context_free.config import MetaConfig

import numpy as np

class DataProcessor:
    def __init__(
        self,
        metaConfig: MetaConfig,
    ):
        self.len_window = int(metaConfig.len_window)
        self.len_source = int(metaConfig.len_source)
        self.Ts = float(metaConfig.Ts)
        self.receiving_timestamps = deque(maxlen=int(self.len_window*self.len_source*1.2))

    def receive_signal(self) -> None:
        self.receiving_timestamps.append(time.time())
        
    def is_ready(self) -> bool:
        return len(self.receiving_timestamps) > 0

    def get_historical_traffic(self) -> np.ndarray:
        current_time = time.time()
        traffic_states = np.zeros(self.len_source, dtype=np.int32)
        
        window_time_bins = np.linspace(
            current_time - self.Ts * self.len_window, 
            current_time - self.Ts * self.len_window * self.len_source, 
            self.len_source 
        )

        # Define time windows going backwards from current time
        # Window i covers time range [window_time_bins[i+1], window_time_bins[i])
        # Iterate through timestamps in reverse (most recent first)
        window_pivot = 0
        for timestamp in reversed(self.receiving_timestamps):
            # Stop if timestamp is older than the oldest window bin
            if timestamp < window_time_bins[-1]:
                break
            elif timestamp < window_time_bins[window_pivot]:
                window_pivot += 1
                continue
            traffic_states[window_pivot] += 1  
        return traffic_states[::-1].copy()
