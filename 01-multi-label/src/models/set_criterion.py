import torch
from models import ModelConfig
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Taken from: 

    """

    def __init__(self, alpha=0.25, gamma=2.5, weights=None, device='cuda', reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.reduction = reduction

        if weights is not None:
            weights = weights.to(device)

        self.loss = nn.BCEWithLogitsLoss(weight=weights)
        self.to(device)

    def forward(self, inputs, targets):
        '''
        Apply focal loss for multi-label classification.
        :param inputs: Tensor of size (batch_size, num_classes) representing logit outputs
        :param targets: Tensor of size (batch_size, num_classes) representing true labels
        :return: Computed focal loss
        '''
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        bce_loss = self.loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


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
        criterion = FocalLoss(device=device, gamma=2.0, weights=class_weights)
    else:
        raise ValueError(f'Invalid loss argument: {model_config.loss_arg}')
    return criterion
