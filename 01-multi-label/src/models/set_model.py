from models import ModelConfig
from torchvision.models import resnet50
import torch
from utils import FileManager


def set_model(model_config: ModelConfig, file_manager: FileManager):
    num_labels = model_config.num_labels

    if model_config.model_arg == 'resnet50':
        model = resnet50(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_labels, bias=True)
        img_size = 224    
    elif model_config.model_arg == 'swin':
        img_size = 224
        raise NotImplementedError
    elif model_config.model_arg == "vit":
        img_size = 224
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid model argument: {model_config.model_arg}')
    return model, img_size
     