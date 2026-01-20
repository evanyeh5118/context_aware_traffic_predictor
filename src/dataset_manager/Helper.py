import numpy as np
from scipy.interpolate import interp1d


def resampleData(timestamps, data, Ts):
    timestamps = np.array(timestamps)
    data = np.array(data)
    
    if len(timestamps) != len(data):
        raise ValueError("timestamps and data must have the same length.")
    
    start_time, end_time = timestamps[0], timestamps[-1]
    new_timestamps = np.arange(start_time, end_time, Ts)
    
    if len(new_timestamps) == 0:
        new_timestamps = np.array([start_time])
    
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    new_data = np.column_stack([
        interp1d(timestamps, data[:, i], kind='linear', fill_value='extrapolate', assume_sorted=False)(new_timestamps)
        for i in range(data.shape[1])
    ])
    
    return new_timestamps, new_data


