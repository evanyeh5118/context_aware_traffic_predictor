"""
Mathematical utility functions.
"""

import numpy as np
import torch
from typing import Union, Tuple


def softmax(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute softmax values for input array.
    
    Args:
        x: Input array (numpy or torch tensor)
        
    Returns:
        Softmax values
    """
    if isinstance(x, torch.Tensor):
        return torch.softmax(x, dim=-1)
    else:
        # Numpy implementation
        e_x = np.exp(x - np.max(x))  # for numerical stability
        return e_x / np.sum(e_x)


def compute_transition_matrix(
    x_t_minus_1: np.ndarray, 
    x_t: np.ndarray, 
    len_window: int, 
    alpha: float = 1e-20
) -> np.ndarray:
    """
    Compute transition matrix from state sequences.
    
    Args:
        x_t_minus_1: Previous states
        x_t: Current states  
        len_window: Window length
        alpha: Smoothing parameter
        
    Returns:
        Transition matrix
    """
    L = len_window + 1
    x_t_minus_1 = x_t_minus_1.astype(int)
    x_t = x_t.astype(int)
    P = np.zeros((L, L))

    for i, j in zip(x_t_minus_1, x_t):
        P[i, j] += 1

    # Normalize rows to get probabilities
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(P, row_sums, where=row_sums != 0)  # avoid division by zero

    P = P + alpha
    P = P / P.sum(axis=1, keepdims=True)

    return P


def generate_balanced_thresholds(arr: np.ndarray, N: int) -> list:
    """
    Generate balanced thresholds to split array into N groups.
    
    Args:
        arr: Input array
        N: Number of groups
        
    Returns:
        List of threshold values
    """
    if N <= 1:
        raise ValueError("N must be greater than 1.")

    # Count the frequency of each unique value
    unique, counts = np.unique(arr, return_counts=True)
    freq_dict = dict(zip(unique, counts))

    # Sorting the unique values by their frequencies
    sorted_values = sorted(freq_dict.keys())
    total_samples = len(arr)

    # Initialize variables for threshold calculation
    thresholds = []
    cum_count = 0
    group_size = total_samples / N
    current_group_count = 0

    # Calculate N-1 thresholds
    for value in sorted_values:
        cum_count += freq_dict[value]
        current_group_count += freq_dict[value]

        # Check if the current group is full
        if current_group_count >= group_size:
            thresholds.append(value)
            current_group_count = 0

        # Adjust the group size dynamically to ensure N-1 thresholds
        remaining_groups = N - len(thresholds) - 1
        if remaining_groups > 0:
            group_size = (total_samples - cum_count) / remaining_groups

        # Stop if we reach exactly N-1 thresholds
        if len(thresholds) == N - 1:
            break

    # Ensure the final threshold list is exactly N-1
    while len(thresholds) < N - 1:
        thresholds.append(sorted_values[-1])

    return thresholds


def assign_groups(arr: np.ndarray, thresholds: list) -> np.ndarray:
    """
    Assign groups based on thresholds.
    
    Args:
        arr: Input array
        thresholds: List of threshold values
        
    Returns:
        Array with group assignments
    """
    group_arr = np.zeros_like(arr)
    for i, thres in enumerate(thresholds):
        group_arr[arr > thres] = i + 1

    return group_arr
