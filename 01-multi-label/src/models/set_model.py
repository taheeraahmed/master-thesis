from models import ModelConfig
from torchvision.models import resnet50, resnet34, alexnet
import torch
from torch import nn
from utils import FileManager


def set_model(model_config: ModelConfig, file_manager: FileManager):
    num_labels = model_config.num_labels

    if model_config.model_arg == 'resnet50':
        model = resnet50(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_labels, bias=True)
        img_size = int(224*2) 
    elif model_config.model_arg == 'resnet34':
        model = resnet34(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_labels, bias=True)
        img_size = int(224*2)     
    elif model_config.model_arg == 'swin':
        img_size = int(224*2) 
        raise NotImplementedError
    elif model_config.model_arg == "vit":
        img_size = int(224*2) 
        raise NotImplementedError
    elif model_config.model_arg == "alexnet": 
        model = alexnet(weights="DEFAULT")
    
        num_target_classes = model_config.num_labels
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 6 * 6, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, num_target_classes),
        )
        img_size = int(224*2) 
    else:
        raise ValueError(f'Invalid model argument: {model_config.model_arg}')
    
    model_info = str(model)
    with open((f"{file_manager.output_folder}/model_architecture.txt"), "w") as file:
        file.write(model_info)
        
    return model, img_size
     