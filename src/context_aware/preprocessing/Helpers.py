from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def interpolateContextData(flags, data, timestamps):  # self.contextDataDpDr -> self.contextDataPorcessed
    return interpolationData(
        np.asarray(flags).astype(int),
        np.asarray(data, dtype=np.float64),
        np.asarray(timestamps),
    )

def normalizeColumns(data: np.ndarray) -> np.ndarray:
    """Normalize columns to [0, 1], handling constant columns safely.

    Ensures output has shape (N, D).
    """
    data_arr = np.asarray(data, dtype=np.float64)
    if data_arr.ndim == 1:
        data_arr = data_arr[..., np.newaxis]
    if data_arr.size == 0:
        return data_arr
    min_values = np.nanmin(data_arr, axis=0)
    max_values = np.nanmax(data_arr, axis=0)
    denom = max_values - min_values
    # Avoid division by zero for constant columns
    denom[denom == 0.0] = 1.0
    normalized = (data_arr - min_values) / denom
    return normalized

def smoothDataByFiltfilt(x, fc, fs, order):
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

def DiscretizedTraffic(data):    
    outputs = []
    for d in data:
        outputs.append(round(d))
    return np.array(outputs)

def FindLastTransmissionIdx(transmission, current_idx):
    while current_idx-1 >= 0:
        current_idx = current_idx-1
        if transmission[current_idx] == 1:
            return current_idx
    return 0

'''
def SmoothFilter(df, fc, fs, order):
    # Ensure the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")
    # Butterworth filter design
    nyquist = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = fc / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    # Apply the filter to each column
    filtered_data = df.apply(lambda col: filtfilt(b, a, col), axis=0)
    # Return the filtered data as a DataFrame with the same index and columns
    return pd.DataFrame(filtered_data, index=df.index, columns=df.columns)
'''