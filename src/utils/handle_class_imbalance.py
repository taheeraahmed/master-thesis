import numpy as np
import torch


def get_class_weights(train_df):
    pos_weights = []

    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                'Pneumonia', 'Pneumothorax']

    for disease in diseases:
        n_positive = np.sum(train_df[disease])
        n_negative = len(train_df) - n_positive

        weight_for_positive = (1 / n_positive) * (len(train_df) / 2.0)

        pos_weights.append(weight_for_positive)

    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float)
    return pos_weights_tensor
