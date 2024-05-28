from models import ModelConfig
import torch


def set_optimizer(model_config: ModelConfig) -> torch.optim.Optimizer:
    model = model_config.model
    optimizer_arg = model_config.optimizer_arg
    if optimizer_arg == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config.learning_rate,
        )
    elif optimizer_arg == 'sgd':
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=model_config.learning_rate,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=True
        )
    elif optimizer_arg == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_config.learning_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_config.learning_rate,
        )
    return optimizer
