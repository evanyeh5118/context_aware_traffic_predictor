import numpy as np
import pandas as pd

from .DeadbandReduction import DataReductionForDataUnit
from .Helper import interpolationData, resampleData, smoothDataByFiltfilt

class DataUnit:
    """Container for context data and associated processing utilities.

    Notes
    - Keeps original attribute names to preserve backwards compatibility.
    - Uses numpy arrays internally where applicable.
    """

    def __init__(self):
        self.name = []
        self.Ts = []
        self.timestamps = []
        self.contextData = []
        self.contextDataDpDr = []
        self.contextDataPorcessed = []
        self.transmitionFlags = []
        self.dimFeatures = []
        self.dataLength = []
        self.compressionRate = []

    def __getitem__(self, key):
        """Return a shallow copy with data sliced by key (e.g., slice or indices)."""
        dataUnitCopy = DataUnit()

        dataUnitCopy.contextData = np.asarray(self.contextData)[key]
        dataUnitCopy.contextDataDpDr = np.asarray(self.contextDataDpDr)[key]
        dataUnitCopy.contextDataPorcessed = np.asarray(self.contextDataPorcessed)[key]
        dataUnitCopy.transmitionFlags = np.asarray(self.transmitionFlags)[key]

        dataUnitCopy.name = self.name
        dataUnitCopy.Ts = self.Ts
        dataUnitCopy.dataLength = (
            dataUnitCopy.contextData.shape[0]
            if hasattr(dataUnitCopy.contextData, "shape")
            else len(dataUnitCopy.contextData)
        )
        denom = len(dataUnitCopy.transmitionFlags) if len(np.asarray(dataUnitCopy.transmitionFlags)) > 0 else 1
        dataUnitCopy.compressionRate = float(np.sum(dataUnitCopy.transmitionFlags)) / float(denom)
        dataUnitCopy.dimFeatures = self.dimFeatures
        base_ts0 = np.asarray(self.timestamps)[0] if len(np.asarray(self.timestamps)) > 0 else 0.0
        dataUnitCopy.timestamps = np.asarray(self.timestamps)[key] - base_ts0
        return dataUnitCopy

    def getContextDataProcessed(self):
        """Return normalized processed context data.

        Handles constant columns safely by avoiding division by zero and
        ensures a 2D array shape (N, D).
        """
        data = np.asarray(self.contextDataPorcessed).copy()
        return self._normalize_columns(data)
    
    def getContextDataProcessedAndSmoothed(self, fc, order):  # self.contextDataPorcessed -> smoothed
        """Smooth processed context data and return its normalized version.

        Parameters
        - fc: cutoff frequency for smoothing
        - order: filter order
        """
        if not self.Ts:
            raise ValueError("Sampling time Ts is not set. Call _resampleContextData() first or set Ts.")
        smoothData = smoothDataByFiltfilt(self.contextDataPorcessed, fc, 1 / float(self.Ts), order)
        return self._normalize_columns(smoothData)

    def getTransmissionFlags(self):
        """Return a copy of transmission flags as a numpy array."""
        return np.asarray(self.transmitionFlags).copy()

    def display(self):
        """Print a concise summary of the data unit."""
        print(
            f"Name: {self.name}, Ts:{self.Ts}, Data length:{self.dataLength}, "
            f"Dim of context:{self.dimFeatures}, Compression rate:{self.compressionRate}"
        )

    def generateTrafficPattern(self, lenWindow):
        """Aggregate transmission flags over fixed windows.

        Parameters
        - lenWindow: window length in samples (int > 0)
        """
        if not isinstance(lenWindow, (int, np.integer)) or lenWindow <= 0:
            raise ValueError("lenWindow must be a positive integer")
        flags = np.asarray(self.transmitionFlags)
        if flags.size == 0:
            return np.array([])
        N_slot = int(np.floor(self.dataLength / lenWindow)) if self.dataLength else int(np.floor(flags.size / lenWindow))
        traffic_state = [
            np.sum(flags[i * lenWindow : (i + 1) * lenWindow])
            for i in range(N_slot)
        ]
        return np.asarray(traffic_state)

    def saveDataUnit(self, filename):
        if not hasattr(self, "contextDataDpDr"):
            raise AttributeError("self.contextDataDpDr is not set. Call _applyDpDr() first.")
        if not hasattr(self, "timestamps"):
            raise AttributeError("self.timestamps is not set.")

        timestamps = np.asarray(self.timestamps)
        data = np.asarray(self.contextDataDpDr)
        transmitionFlags = np.asarray(self.transmitionFlags)
        print(self.name,timestamps.shape, data.shape, transmitionFlags.shape)

        if data.shape[0] != timestamps.shape[0]:
            raise ValueError("timestamps and contextDataDpDr must have the same number of rows.")

        df = pd.DataFrame(data)
        df.insert(0, "Time", timestamps)
        df.insert(1, "Transmition Flags", transmitionFlags)
        df.to_csv(filename, index=False)
    
    def _setContextData(self, contextData):
        """Set raw context data and update shape-related metadata."""
        context_arr = np.asarray(contextData)
        if context_arr.ndim == 1:
            context_arr = context_arr[..., np.newaxis]
        self.contextData = context_arr
        self.dataLength = int(context_arr.shape[0])
        self.dimFeatures = int(context_arr.shape[1])
        
    def _applyDpDr(self, dbParameter=0.01, alpha=0.01, mode="fixed"):  # self.contextData -> self.contextDataDpDr
        contextDataDpDr, transmitionFlags = DataReductionForDataUnit(
            self, dbParameter=dbParameter, alpha=alpha, mode=mode
        )
        self.contextDataDpDr = contextDataDpDr
        self.transmitionFlags = transmitionFlags
        denom = self.transmitionFlags.shape[0] if hasattr(self.transmitionFlags, "shape") else len(self.transmitionFlags)
        denom = denom if denom > 0 else 1
        self.compressionRate = float(np.sum(self.transmitionFlags)) / float(denom)

    def _interpolateCotextAfterDpDr(self):  # self.contextDataDpDr -> self.contextDataPorcessed
        self.contextDataPorcessed = interpolationData(
            np.asarray(self.transmitionFlags).astype(int),
            np.asarray(self.contextDataDpDr, dtype=np.float64),
            np.asarray(self.timestamps),
        )

    def _resampleContextData(self):  # self.contextData -> self.contextData
        ts_arr = np.asarray(self.timestamps)
        if ts_arr.size < 2:
            raise ValueError("Not enough timestamps to infer sampling time Ts.")
        self.Ts = round(float(np.mean(ts_arr[1:] - ts_arr[:-1])), 2)
        (_, self.contextData) = resampleData(ts_arr, self.contextData, self.Ts)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    @staticmethod
    def _normalize_columns(data: np.ndarray) -> np.ndarray:
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

