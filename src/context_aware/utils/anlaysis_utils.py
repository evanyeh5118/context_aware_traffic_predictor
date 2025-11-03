import re
import torch.nn as nn
import torch
import numpy as np


def compute_f1_scores(predictions: np.ndarray, ground_truth: np.ndarray, num_classes: int):
    """
    Computes the F1-Score for each class in a multi-class classification problem.

    Parameters:
    - predictions (np.ndarray): Array of predicted class labels.
    - ground_truth (np.ndarray): Array of actual class labels.
    - num_classes (int): Total number of classes.

    Returns:
    - f1_scores (np.ndarray): Array of F1-Score values for each class.
    """
    # Initialize arrays for True Positives, False Positives, False Negatives
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)

    # Calculate TP, FP, FN for each class
    for i in range(num_classes):
        TP[i] = np.sum((predictions == i) & (ground_truth == i))
        FP[i] = np.sum((predictions == i) & (ground_truth != i))
        FN[i] = np.sum((predictions != i) & (ground_truth == i))

    # Calculate Precision, Recall, F1-Score for each class
    precision = TP / (TP + FP + 1e-10)  # Adding epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    return f1_scores

def compute_weighted_f1_score(predictions: np.ndarray, ground_truth: np.ndarray, num_classes: int):
    """
    Computes the Weighted-Averaged F1-Score for a multi-class classification problem.

    Parameters:
    - predictions (np.ndarray): Array of predicted class labels.
    - ground_truth (np.ndarray): Array of actual class labels.
    - num_classes (int): Total number of classes.

    Returns:
    - weighted_f1 (float): The Weighted-Averaged F1-Score.
    """
    # Initialize arrays for True Positives, False Positives, False Negatives
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)

    # Calculate TP, FP, FN for each class
    for i in range(num_classes):
        TP[i] = np.sum((predictions == i) & (ground_truth == i))
        FP[i] = np.sum((predictions == i) & (ground_truth != i))
        FN[i] = np.sum((predictions != i) & (ground_truth == i))

    # Calculate Precision, Recall, F1-Score for each class
    precision = TP / (TP + FP + 1e-10)  # Adding epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Calculate weighted average
    class_counts = np.array([np.sum(ground_truth == i) for i in range(num_classes)])
    weighted_f1 = np.sum(f1_scores * class_counts) / np.sum(class_counts)

    return weighted_f1




