from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np


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