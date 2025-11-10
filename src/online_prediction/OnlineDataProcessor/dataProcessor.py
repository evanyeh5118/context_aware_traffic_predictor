import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

# Reuse the same helpers used in offline preprocessing / relay predictor
from src.context_aware.preprocessing.Helpers import (
    interpolationData,
    normalizeColumns
)
from src.context_aware.preprocessing.filter import MultiDimExpSmoother

class DataProcessor:
    def __init__(
        self,
        config,
    ):
        self.window_length = int(config.window_length)
        self.smooth_fc = float(config.smooth_fc)
        self.smooth_order = int(config.smooth_order)
        self.Ts = float(config.Ts)
        self.smooth_fs = 1.0 / self.Ts
        self.min_vals = config.min_vals #shape: (num_features,)
        self.max_vals = config.max_vals #shape: (num_features,)
        self.denom = self.max_vals - self.min_vals
        self.dim_data = config.dim_data
        self.filter = MultiDimExpSmoother(fc=self.smooth_fc, Ts=self.Ts, buffer_size=500)

        self._context_buffer = deque(maxlen=self.window_length)
        self._timestamp_buffer = deque(maxlen=self.window_length)
        self._last_window_context = np.zeros(self.dim_data)


    def add_data_point(
        self,
        context_data: np.ndarray,
    ) -> None:

        context_arr = np.asarray(context_data, dtype=np.float64)
        if context_arr.ndim > 1:
            context_arr = context_arr.reshape(-1)

        self._context_buffer.append(context_arr.copy())
        self._timestamp_buffer.append(time.time())

    def is_ready(self) -> bool:
        return len(self._context_buffer) >= self.window_length

    def get_window_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # Prepare context array
        context = np.asarray(list(self._context_buffer), dtype=np.float64)
        context_timestamps = np.asarray(list(self._timestamp_buffer), dtype=np.float64)

        # Build time grid ending at current time
        t_end = time.time()
        t_start = t_end - self.Ts * (self.window_length-1)

        # 1) Create bins
        context_bin = np.zeros((self.window_length, context.shape[1]), dtype=np.float64)
        timestamps_bin = np.linspace(t_start, t_end, self.window_length, dtype=np.float64)
        flags = np.full(self.window_length, 0, dtype=np.int32)

        # 2) Place each sample into its bin (use latest sample if multiple in a bin)
        for s_t, s_x in zip(context_timestamps, context):
            idx = int(np.floor((s_t - t_start ) / self.Ts))
            if 0 <= idx < self.window_length:
                if s_t > timestamps_bin[idx]:
                    context_bin[idx] = s_x
                    flags[idx] = 1

        # 3) Forward-fill: for each bin without a new sample, use the previous bin's value
        for idx in range(0, self.window_length):
            if flags[idx] == 0:  # No new sample in this bin
                if idx > 0:
                    context_bin[idx,:] = context_bin[idx - 1,:]
                else:
                    context_bin[idx,:] = self._last_window_context
        
        self._last_window_context = context_bin[-1,:].copy()

        context_bin = np.array(context_bin, dtype=np.float64)
        flags = np.array(flags, dtype=np.int32)
        
        return context_bin, flags, timestamps_bin

    def get_window_features(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        context_no_smooth, flags, timestamps_bin = self.get_window_data()

        context = interpolationData(flags, context_no_smooth, timestamps_bin)
        context = self.filter.filter(context)
        context = normalizeColumns(context, self.max_vals, self.min_vals)
        last_trans_sources = self._last_window_context.copy()

        if last_trans_sources.ndim == 1:
            last_trans_sources = last_trans_sources.reshape(1, -1)
    
        return context, last_trans_sources, context_no_smooth, (flags, timestamps_bin)

