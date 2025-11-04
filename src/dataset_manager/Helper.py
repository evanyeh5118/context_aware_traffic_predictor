import numpy as np
from scipy.interpolate import interp1d


def resampleData(timestamps, data, Ts):
    timestamps = np.array(timestamps)
    data = np.array(data)
    
    # Ensure timestamps and data have matching lengths
    if len(timestamps) != len(data):
        raise ValueError("timestamps and data must have the same length.")
    
    # Define target number of samples to match input size
    num_samples = len(timestamps)
    
    # Define the new timestamps ensuring the same length as input
    start_time, end_time = timestamps[0], timestamps[-1]
    new_timestamps = np.linspace(start_time, end_time, num_samples)
    
    # Ensure data is at least 2D
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    # Interpolate each dimension separately
    new_data = np.column_stack([
        interp1d(timestamps, data[:, i], kind='linear', fill_value='extrapolate', assume_sorted=False)(new_timestamps)
        for i in range(data.shape[1])
    ])
    
    return new_timestamps, new_data


