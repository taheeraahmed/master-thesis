import torch
from models import ModelConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Taken from: https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch/blob/master/losses.py
    """
    def __init__(self, device, gamma = 1.0):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype = torch.float32).to(device)
        self.eps = 1e-6
        
    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').to(self.device)
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss =  (1-pt)**self.gamma * BCE_loss
        
        return F_loss.mean() 

def set_criterion(model_config: ModelConfig, class_weights: torch.Tensor = None) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_config.loss_arg == 'mlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    elif model_config.loss_arg == 'wmlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    elif model_config.loss_arg == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(weight=None)
    elif model_config.loss_arg == 'wbce':
        raise NotImplementedError
    elif model_config.loss_arg == 'focal':
        criterion = FocalLoss(device=device, gamma=2.0)
    else:
        raise ValueError(f'Invalid loss argument: {model_config.loss_arg}')
    return criterion