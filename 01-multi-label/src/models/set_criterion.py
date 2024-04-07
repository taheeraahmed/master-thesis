import torch
from models import ModelConfig

def set_criterion(model_config: ModelConfig, class_weights: torch.Tensor = None) -> torch.nn.Module:
    if model_config.loss_arg == 'mlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    elif model_config.loss_arg == 'wmlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    elif model_config.loss_arg == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss(weight=None)
    elif model_config.loss_arg == 'wbce':
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid loss argument: {model_config.loss_arg}')
    return criterion