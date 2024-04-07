import torch
from models import ModelConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Taken from: https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch/blob/master/losses.py
    Modified to include class weights.
    """
    def __init__(self, device, gamma=1.0, class_weights=None):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device)
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.eps = 1e-6
        
    def forward(self, input, target):
        if self.class_weights is not None:
            BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none', pos_weight=self.class_weights).to(self.device)
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').to(self.device)
        
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        
        return F_loss.mean()


def set_criterion(model_config: ModelConfig, class_weights: torch.Tensor = None) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_config.loss_arg == 'mlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    elif model_config.loss_arg == 'wmlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    elif model_config.loss_arg == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif model_config.loss_arg == 'wbce':
        criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    elif model_config.loss_arg == 'focal':
        criterion = FocalLoss(device=device, gamma=2.0)
    elif model_config.loss_arg == 'wfocal':
        criterion = FocalLoss(device=device, gamma=2.0, class_weights=class_weights)
    else:
        raise ValueError(f'Invalid loss argument: {model_config.loss_arg}')
    return criterion