import torch

def _create_matrix(dt: float, L: int, degree: int) -> torch.Tensor:
    n_values = torch.arange(L, dtype=torch.float32)
    matrix = torch.stack([(n_values * dt) ** d for d in range(degree + 1)], dim=1)
    return matrix

def _compute_poly_matrix(len_source, len_target, dt, degree, device):
    # Create matrices for source and prediction
    A_s = _create_matrix(dt, len_source, degree)  # Shape: (len_source, degree + 1)
    A_p = _create_matrix(dt, len_source + len_target, degree)[len_source:]  # Shape: (len_target, degree + 1)
    A_s = A_s.to(device)
    A_p = A_p.to(device)

    A_s_t = A_s.T
    M = A_p @ torch.linalg.inv(A_s_t @ A_s) @ A_s_t
    return M

def _compute_poly_matrix_regularized(len_source, len_target, dt, degree, device, penalty=1e-4):
    """
    Compute the polynomial projection matrix with a regularized inverse.
    
    Args:
        len_source (int): Number of source points.
        len_target (int): Number of target points.
        dt (float): Time step or spacing.
        degree (int): Degree of polynomial basis.
        device (torch.device): Target device for computation.
        penalty (float): Regularization strength (λ). Default = 1e-4.
    """
    # Build polynomial design matrices
    A_s = _create_matrix(dt, len_source, degree)  # (len_source, degree + 1)
    A_p = _create_matrix(dt, len_source + len_target, degree)[len_source:]  # (len_target, degree + 1)
    A_s = A_s.to(device)
    A_p = A_p.to(device)

    # Compute regularized inverse: (A_s^T A_s + λI)^(-1)
    A_s_t = A_s.T
    I = torch.eye(A_s_t.shape[0], device=device)
    M = A_p @ torch.linalg.inv(A_s_t @ A_s + penalty * I) @ A_s_t

    return M

def _compute_feature_length(data_length):
    if data_length < 2:
        raise ValueError("Data length must be at least 2 to compute pairwise distances.")
    # Magnitudes + Pairwise distances
    feature_length = data_length + (data_length * (data_length - 1)) // 2
    return feature_length
