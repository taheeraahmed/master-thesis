from models import ModelConfig
from torchvision.models import resnet50, resnet34, alexnet
import torch
from torch import nn
from utils import FileManager


def set_model(model_arg: str, num_labels: int):

    if model_arg == 'resnet50':
        model = resnet50(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_labels, bias=True)
        img_size = int(224*2) 
    elif model_arg == 'resnet34':
        model = resnet34(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_labels, bias=True)
        img_size = int(224*2)     
    elif model_arg == 'swin':
        img_size = int(224*2) 
        raise NotImplementedError
    elif model_arg == "vit":
        img_size = int(224*2) 
        raise NotImplementedError
    elif model_arg == "alexnet": 
        model = alexnet(weights="DEFAULT")
    
        num_target_classes = num_labels
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 6 * 6, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, num_target_classes),
        )
        img_size = int(224*2) 
    else:
        raise ValueError(f'Invalid model argument: {model_arg}')
        
    return model, img_size
     