import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

# Reuse the same helpers used in offline preprocessing / relay predictor
from src.context_aware.preprocessing.Helpers import (
    smoothDataByFiltfilt,
)


class DataProcessor:
    """
    Online windowed processor that receives streaming context data and
    computes inputs required by the model: `sources`, `last_trans_sources`,
    and `sourcesNoSmooth` for the current window.

    The computations mirror the notebook/main pipeline:
    - Interpolate by transmission flags
    - Smooth with filtfilt
    - Normalize per-window to [0, 1]
    - Extract last transmitted context within the window
    """

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
    
        self._context_buffer = deque(maxlen=self.window_length * 2)
        self._timestamp_buffer = deque(maxlen=self.window_length * 2)
        self._last_trans_sources: Optional[np.ndarray] = None

    def add_data_point(
        self,
        context_data: np.ndarray,
    ) -> None:

        context_arr = np.asarray(context_data, dtype=np.float64)
        if context_arr.ndim > 1:
            context_arr = context_arr.reshape(-1)

        self._context_buffer.append(context_arr.copy())
        self._timestamp_buffer.append(time.time())
        # Online: all flags are 1; we don't need to store or update anything else here

    def _have_full_window(self) -> bool:
        return len(self._context_buffer) >= self.window_length

    def get_window_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(self._context_buffer) == 0:
            return None

        # Prepare context array
        context = np.asarray(list(self._context_buffer), dtype=np.float64)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        context_timestamps = np.asarray(list(self._timestamp_buffer), dtype=np.float64)

        # Build time grid ending at current time
        t_end = time.time()
        t_start = t_end - self.Ts * (self.window_length - 1)

        # 1) Create bins
        context_bin = np.zeros((self.window_length, context.shape[1]), dtype=np.float64)
        bin_timestamps = np.full(self.window_length, -np.inf, dtype=np.float64)
        flags = np.full(self.window_length, 0, dtype=np.int32)

        # 2) Place each sample into its bin (use latest sample if multiple in a bin)
        for s_t, s_x in zip(context_timestamps, context):
            idx = int(np.floor((s_t - t_start ) / self.Ts))
            if 0 <= idx < self.window_length:
                if s_t > bin_timestamps[idx]:
                    context_bin[idx] = s_x
                    bin_timestamps[idx] = s_t
                    flags[idx] = 1

        # 3) Forward-fill: for each bin without a new sample, use the previous bin's value
        for idx in range(0, self.window_length):
            if flags[idx] == 0:  # No new sample in this bin
                if idx > 0:
                    context_bin[idx] = context_bin[idx - 1]
                else:
                    context_bin[idx] = self._last_trans_sources
        
        packet_count = np.sum(flags)

        self._last_trans_sources = context_bin[-1:]
        return context_bin, packet_count

    def get_window_features(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not self._have_full_window():
            return None

        context_no_smooth = self.get_window_data()

        context_smoothed = smoothDataByFiltfilt(
            context_no_smooth,     
            self.smooth_fc,
            self.smooth_fs,
            self.smooth_order,
        )
        denom = self.max_vals - self.min_vals
        denom[denom == 0.0] = 1.0
        context = (context_smoothed - self.min_vals) / denom

        last_trans_sources = self._last_trans_sources.copy()
    
        return context, last_trans_sources, context_no_smooth

