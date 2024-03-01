import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch

def calculate_class_weights(integer_labels):
    """
    Calculate class weights given a series of integer labels.
    """
    unique_classes = np.unique(integer_labels)
    class_weights = compute_class_weight(
        'balanced', classes=unique_classes, y=integer_labels)
    return torch.tensor(class_weights, dtype=torch.float)
