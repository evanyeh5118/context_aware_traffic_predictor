import numpy as np

def poly_fit_smoother(data: np.ndarray, degree: int = 3) -> np.ndarray:
    # Ensure numpy array and 2D
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]

    L, D = data.shape

    # x-axis: using index 0, 1, ..., L-1
    x = np.arange(L, dtype=np.float64)

    fitted = np.empty_like(data)

    for d in range(D):
        # Fit polynomial for column d
        coeffs = np.polyfit(x, data[:, d], degree)
        # Evaluate polynomial at x
        fitted[:, d] = np.polyval(coeffs, x)

    return fitted


class OnlineGainOptimizer:
    def __init__(self, gain_init=0.0, lr=0.01, gain_min=None, gain_max=None):
        self.gain = float(gain_init)
        self.lr = float(lr)
        self.gain_min = gain_min
        self.gain_max = gain_max

    def update(self, x, x_):
        x = np.asarray(x, dtype=float)
        x_ = np.asarray(x_, dtype=float)

        if x.shape != x_.shape:
            raise ValueError(f"x and x_ must have the same shape, got {x.shape} vs {x_.shape}")

        # Error: e = k * x_ - x
        error = self.gain * x_ - x

        # Gradient of mean loss w.r.t. k:
        # grad = 2 * mean( x_ * (k * x_ - x) )
        grad = 2.0 * np.mean(x_ * error)

        # Gradient descent step
        self.gain -= self.lr * grad

        # Optional projection to [k_min, k_max]
        if self.gain_min is not None:
            self.gain = max(self.gain_min, self.gain)
        if self.gain_max is not None:
            self.gain = min(self.gain_max, self.gain)

        return self.gain

    def get_gain(self):
        return self.gain
