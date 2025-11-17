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


import numpy as np

class KalmanFilter:
    def __init__(self, dim, q_factor=1e-5, r_factor=1e-1):
        """
        dim : int
            Dimension of the state and observation.
        F : (dim, dim)
            State transition matrix.
        Q : (dim, dim)
            Process noise covariance.
        H : (dim, dim)
            Observation matrix.
        R : (dim, dim)
            Observation noise covariance.
        x0 : (dim,)
            Initial state mean.
        P0 : (dim, dim)
            Initial state covariance.
        """
        self.dim = dim
        # Default values
        self.F = np.eye(dim)
        self.Q = q_factor* np.eye(dim)
        self.H = np.eye(dim)
        self.R = r_factor * np.eye(dim)
        
        self.x = np.zeros(dim)
        self.P = np.eye(dim)

    def predict(self):
        """Time update (prediction)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        """Measurement update (correction).

        z : (dim,)
            Observation at current time step.
        """
        z = np.asarray(z)
        
        # Innovation (residual)
        y = z - (self.H @ self.x)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Updated state estimate
        self.x = self.x + K @ y
        
        # Updated covariance
        I = np.eye(self.dim)
        self.P = (I - K @ self.H) @ self.P
        return self.x, self.P

    def filter(self, data, previous=None):
        """
        Apply the Kalman filter to a batch of data, optionally continuing from previous data.

        Parameters
        ----------
        data : ndarray (timesteps, features)
            The observations to filter.
        previous : ndarray (timesteps_prev, features), optional
            Previous batches of data to continue the filtering from.

        Returns
        -------
        filtered : ndarray (timesteps, features)
            The filtered estimates for the provided data.
        """

        if data is None or len(data) == 0:
            return np.array([]).reshape(0, self.dim)

        if previous is not None and len(previous) > 0:
            data_full = np.concatenate([previous, data], axis=0)
        else:
            data_full = np.array(data, copy=True)

        filtered_full = []

        for t in range(data_full.shape[0]):
            self.predict()
            self.update(data_full[t])
            # Avoid .copy() to enforce no reference option; depends on requirements
            filtered_full.append(self.x.copy())

        filtered_full = np.stack(filtered_full, axis=0)
        filtered_full = filtered_full[-data.shape[0]:, :]
        return filtered_full - (np.mean(filtered_full, axis=0) - np.mean(data, axis=0))


class TikhonovSmoother:
    def __init__(self, dim, lam=1.0, dt=1.0):
        self.dim = dim
        self.lam = lam
        self.dt = dt

    def smooth(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        N, D = data.shape
        smoothed_data = np.empty((N, D), dtype=np.float64)
        for d in range(D):
            x_smooth, _ = self._smooth_one_dim(data[:, d])
            smoothed_data[:, d] = x_smooth
        return smoothed_data

    def _smooth_one_dim(self, x):
        x = np.asarray(x, dtype=np.float64)
        N = x.shape[0]

        # Anchor at the original starting point
        x0 = x[0]
        x_tilde = x - x0

        # --- Build integration matrix C (N x N) ---
        # (Cv)[i] = dt * sum_{j=0}^{i-1} v[j], with (Cv)[0] = 0
        C = np.tril(np.ones((N, N), dtype=np.float64), k=-1) * self.dt

        # --- Build first-difference matrix D ((N-1) x N) ---
        D = np.zeros((N - 1, N), dtype=np.float64)
        for i in range(N - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0

        # --- Solve (C^T C + lam D^T D) v = C^T x_tilde ---
        A = C.T @ C + self.lam * (D.T @ D)
        b = C.T @ x_tilde

        v_smooth = np.linalg.solve(A, b)

        # --- Reconstruct x_smooth from v_smooth ---
        x_smooth = x0 + C @ v_smooth

        return x_smooth, v_smooth
