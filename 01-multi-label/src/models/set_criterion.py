import torch
from models import ModelConfig

def set_criterion(model_config: ModelConfig, class_weights: torch.Tensor = None) -> torch.nn.Module:
    if model_config.loss_arg == 'mlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    elif model_config.loss_arg == 'wmlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    elif model_config.loss_arg == 'bce':
        """
        Get this error for both bce and wbce with resnet:
        RuntimeError: CUDA error: device-side assert triggered
        CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
        For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
        Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
        """
        criterion = torch.nn.BCELoss()
        raise NotImplementedError
    elif model_config.loss_arg == 'wbce':
        criterion = torch.nn.BCELoss(weight=class_weights)
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid loss argument: {model_config.loss_arg}')
    return criterion