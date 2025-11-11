import numpy as np


class MultiDimExpSmoother:
    """
    Multi-dimensional exponential smoother (first-order IIR).

    - Causal, stable, no filtfilt edge artifacts.
    - Keeps smoothed data close to input (controlled by fc).
    - Input: (T, D) or (T,)
    - Output: same shape.
    """

    def __init__(self, fc: float, Ts: float, buffer_size: int = 1000):
        """
        Parameters
        ----------
        fc : float
            Approximate cutoff frequency (Hz) for smoothing.
            Larger fc => less smoothing (closer to raw).
        Ts : float
            Sampling period (seconds).
        buffer_size : int
            Number of past *raw* samples to keep in history (optional).
        """
        self.fc = float(fc)
        self.Ts = float(Ts)
        self.buffer_size = int(buffer_size)

        # Map cutoff to time constant tau ~ 1/(2*pi*fc)
        # and then to alpha = exp(-Ts / tau).
        # y[n] = (1 - alpha) * x[n] + alpha * y[n-1]
        if self.fc <= 0:
            raise ValueError("fc must be > 0")

        tau = 1.0 / (2.0 * np.pi * self.fc)
        self.alpha = float(np.exp(-self.Ts / tau))

        # Internal state: last output sample per dimension
        self._y_prev = None  # shape (D,)

        # Optional raw history, if you want to inspect later
        self.history = None  # shape (<=buffer_size, D)

    def reset(self):
        """Reset internal state and history."""
        self._y_prev = None
        self.history = None

    def _ensure_state(self, dim: int):
        """Init previous output to zeros if first call."""
        if self._y_prev is None or self._y_prev.shape[0] != dim:
            self._y_prev = np.zeros(dim, dtype=float)

    def filter(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data, dtype=float)
        was_1d = False
        if x.ndim == 1:
            x = x[:, np.newaxis]
            was_1d = True

        T, D = x.shape
        self._ensure_state(D)

        # Update raw history (optional)
        if self.history is None:
            self.history = x.copy()
        else:
            self.history = np.vstack([self.history, x])
            if self.history.shape[0] > self.buffer_size:
                self.history = self.history[-self.buffer_size :, :]

        y = np.empty_like(x)
        alpha = self.alpha
        y_prev = self._y_prev

        # Causal exponential smoothing
        for n in range(T):
            x_n = x[n, :]        # (D,)
            y_n = (1.0 - alpha) * x_n + alpha * y_prev
            y[n, :] = y_n
            y_prev = y_n

        self._y_prev = y_prev

        if was_1d:
            return y[:, 0]
        return y


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian  # <-- fixed import

# === The chunk smoother from Option B ===
class ChunkSmoother:
    def __init__(self, dim: int, kernel = None):
        if kernel is None:
            kernel = gaussian(100, std=1.0)
        self.kernel = kernel / kernel.sum()
        self.W = len(kernel)
        self.H = (self.W - 1) // 2
        self.history_tail = None
        self.dim = dim

    def process(self, x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=np.float64)
        if x_new.ndim == 1:
            x_new = x_new[:, None]
        L, D = x_new.shape

        assert D == self.dim

        if self.history_tail is None:
            x_ext = x_new
        else:
            x_ext = np.vstack([self.history_tail, x_new])

        y_ext = np.empty_like(x_ext)
        for d in range(D):
            y_ext[:, d] = np.convolve(x_ext[:, d], self.kernel, mode="same")

        if self.history_tail is None:
            y_new = y_ext
        else:
            y_new = y_ext[self.history_tail.shape[0]:]

        self.history_tail = x_new[-min(L, self.H):].copy()
        return y_new
