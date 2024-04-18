from models import ModelConfig
from utils import FileManager
import torch

def set_optimizer(model_config: ModelConfig, file_manager: FileManager) -> torch.optim.Optimizer:
    model = model_config.model
    optimizer_arg = model_config.optimizer_arg
    if optimizer_arg == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config.learning_rate,
            betas=(0.9, 0.999),
        )
    elif optimizer_arg == 'sgd':
        optimizer = torch.optim.SGD(
            # TODO ONLY LAST LAYER
            #model.fc.parameters(),
            model.parameters(),
            lr=model_config.learning_rate,
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
        
    file_manager.logger.info(f"Using optimizer: {optimizer}")
    return optimizer