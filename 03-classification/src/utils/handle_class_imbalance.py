import numpy as np
import torch


def get_class_weights(train_df, labels):
    """
    Calculate class weights for imbalanced dataset

    :param train_df: dataframe with training data
    :param labels: list of labels
    
    TODO: This won't work if you have a multi-label dataset, must calculate negative weights as well
    """
    pos_weights = []
    print('Calculating class weights')
    print(labels)
    for disease in labels:
        n_positive = np.sum(train_df[disease])
        weight_for_positive = (1 / n_positive) * (len(train_df) / 2.0)
        pos_weights.append(weight_for_positive)
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float)

    assert len(pos_weights) == len(labels)
    return pos_weights_tensor
