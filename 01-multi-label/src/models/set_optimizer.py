from models import ModelConfig
from utils import FileManager
import torch

def get_classifying_head(model_config: ModelConfig) -> torch.nn.Module:
    model_arg = model_config.model_arg
    model = model_config.model

    if model_arg == 'resnet50':
        classifying_head = model.fc.parameters()
    elif model_arg == 'alexnet' or 'densenet121' or 'efficientnet':
        classifying_head = model.classifier.parameters()
    elif model_arg == 'vit':
        classifying_head = model.heads.parameters()

    return classifying_head
    



def set_optimizer(model_config: ModelConfig, file_manager: FileManager) -> torch.optim.Optimizer:
    model = model_config.model
    optimizer_arg = model_config.optimizer_arg

    # step 1: TODO COMMENT OUT
    classifying_head = get_classifying_head(model_config)

    if optimizer_arg == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config.learning_rate,
            betas=(0.9, 0.999),
        )
    elif optimizer_arg == 'sgd':
        optimizer = torch.optim.SGD(
            # step 1: TODO ONLY LAST LAYER
            classifying_head,
            #model.parameters(),
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