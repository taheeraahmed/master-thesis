import torch
from models import ModelConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Taken from: 
    
    """
    def __init__(self, alpha=0.25, gamma=2.5, weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss = nn.BCEWithLogitsLoss(weight=weights)
    def forward(self, inputs, targets):
        '''
        :param inputs: batch_size * dim
        :param targets: (batch,)
        :return:
        '''
        bce_loss = self.loss(inputs, targets)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss

def set_criterion(model_config: ModelConfig, class_weights: torch.Tensor = None) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_config.loss_arg == 'mlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    elif model_config.loss_arg == 'wmlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
        raise NotImplementedError
    elif model_config.loss_arg == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif model_config.loss_arg == 'wbce':
        criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
        raise NotImplementedError
    elif model_config.loss_arg == 'focal':
        criterion = FocalLoss(device=device, gamma=2.0)
        raise NotImplementedError
    elif model_config.loss_arg == 'wfocal':
        criterion = FocalLoss(device=device, gamma=2.0, weights=class_weights)
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid loss argument: {model_config.loss_arg}')
    return criterion