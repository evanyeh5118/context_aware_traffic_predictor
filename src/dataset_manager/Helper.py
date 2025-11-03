import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

def interpolationData(flags, data, time):
    if data.ndim == 1:
        data = data[:, np.newaxis]  # Convert (T,) -> (T, 1)
    
    T, N = data.shape  

    valid_indices = flags == 1  # Where flags are 0 (valid data)
    flagged_indices = flags == 0  # Where flags are 1 (to be interpolated)

    valid_time = time[valid_indices]
    valid_data = data[valid_indices]  # Shape: (num_valid, N)
    interpolated_data = np.copy(data)
    for i in range(N):
        interp_func = interp1d(valid_time, valid_data[:, i], kind='linear', fill_value="extrapolate")
        interpolated_data[flagged_indices, i] = interp_func(time[flagged_indices])

    if interpolated_data.shape[1] == 1:
        interpolated_data = interpolated_data.flatten()

    return interpolated_data


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


def smoothDataByFiltfilt(x, fc, fs, order):
    """
    Applies a Butterworth low-pass filter to a NumPy array.

    Parameters:
    x (np.ndarray): Input data of shape (len_data, dim).
    fc (float): Cutoff frequency of the filter.
    fs (float): Sampling frequency.
    order (int): Order of the Butterworth filter.

    Returns:
    np.ndarray: Filtered data of the same shape as x.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if x.ndim != 2:
        x = x.reshape(-1, 1)
        #raise ValueError("Input array must be 2D with shape (len_data, dim).")
    
    # Butterworth filter design
    nyquist = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = fc / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Apply the filter along axis 0 (time dimension)
    filtered_data = np.apply_along_axis(lambda col: filtfilt(b, a, col), axis=0, arr=x)
    
    return filtered_data