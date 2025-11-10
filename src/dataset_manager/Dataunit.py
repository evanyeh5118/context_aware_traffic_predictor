import numpy as np
import pandas as pd

from .DeadbandReduction import DataReductionForDataUnit
from .Helper import resampleData

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
        self.transmitionFlags = []
        self.dimFeatures = []
        self.dataLength = []
        self.compressionRate = []
        self.lenWindow = 0

    def __getitem__(self, key):
        """Return a shallow copy with data sliced by key (e.g., slice or indices)."""
        dataUnitCopy = DataUnit()

        dataUnitCopy.lenWindow = self.lenWindow
        dataUnitCopy.contextData = np.asarray(self.contextData)[key]
        dataUnitCopy.contextDataDpDr = np.asarray(self.contextDataDpDr)[key]
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

    def split(self, split_ratio: float):
        if self.contextData is None:
            raise ValueError("Context data is not set")
        first_size = int(split_ratio*len(self.contextData))
        first = self[:first_size]
        second = self[first_size:]
        return first, second

    def getMaxMinMbnVals(self):
        return np.max(self.contextData, axis=0), np.min(self.contextData, axis=0)

    def getContextData(self):
        """Return a copy of context data as a numpy array."""
        return np.asarray(self.contextData).copy()

    def getContextDataDpDr(self):
        """Return a copy of context data after deadband reduction as a numpy array."""
        return np.asarray(self.contextDataDpDr).copy()

    def getTimestamps(self):
        """Return a copy of timestamps as a numpy array."""
        return np.asarray(self.timestamps).copy()

    def getTransmissionFlags(self):
        """Return a copy of transmission flags as a numpy array."""
        return np.asarray(self.transmitionFlags).copy()

    def display(self):
        """Display a summary of the data unit's properties."""
        print("========= Data Unit Summary =========")
        print(f"  Name:                {self.name}")
        print(f"  Length Window:        {self.lenWindow}")
        print(f"  Sampling Interval Ts: {self.Ts}")
        print(f"  Data Length:          {self.dataLength}")
        print(f"  Context Dimension:    {self.dimFeatures}")
        print(f"  Compression Rate:     {self.compressionRate:.4f}")
        print("=====================================")

    def generateTrafficPattern(self):
        """Aggregate transmission flags over fixed windows.

        Parameters
        - lenWindow: window length in samples (int > 0)
        """
        lenWindow = self.lenWindow
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
        contextData_arr = np.asarray(self.contextDataDpDr)
        timestamps_arr = np.asarray(self.timestamps)
        transmission_flags_arr = np.asarray(self.transmitionFlags)

        if not (len(contextData_arr) == len(timestamps_arr) == len(transmission_flags_arr)):
            raise ValueError(
                f"Length mismatch: contextDataDpDr({len(contextData_arr)}), "
                f"timestamps({len(timestamps_arr)}), "
                f"transmitionFlags({len(transmission_flags_arr)})"
            )
        df = pd.DataFrame(contextData_arr)
        df.insert(0, "Time", timestamps_arr)
        df.insert(1, "Transmition Flags", transmission_flags_arr)
        df.to_csv(filename, index=False)

    # ====================================================================
    # ===================== For Dataset Processing =======================
    # ====================================================================
    def setContextData(self, contextData):
        """Set raw context data and update shape-related metadata."""
        context_arr = np.asarray(contextData)
        if context_arr.ndim == 1:
            context_arr = context_arr[..., np.newaxis]
        self.contextData = context_arr
        self.dataLength = int(context_arr.shape[0])
        self.dimFeatures = int(context_arr.shape[1])
        
    def resampleContextData(self):  # self.contextData -> self.contextData
        ts_arr = np.asarray(self.timestamps)
        if ts_arr.size < 2:
            raise ValueError("Not enough timestamps to infer sampling time Ts.")
        self.Ts = round(float(np.mean(ts_arr[1:] - ts_arr[:-1])), 2)
        (_, self.contextData) = resampleData(ts_arr, self.contextData, self.Ts)

    def applyDpDr(self, dbParameter=0.01, alpha=0.01, mode="fixed"):  # self.contextData -> self.contextDataDpDr
        self.contextDataDpDr, self.transmitionFlags = DataReductionForDataUnit(
            self, dbParameter=dbParameter, alpha=alpha, mode=mode
        )
        self.compressionRate = float(np.sum(self.transmitionFlags)) / float(len(self.transmitionFlags))


    

