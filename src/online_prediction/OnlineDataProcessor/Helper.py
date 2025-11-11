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