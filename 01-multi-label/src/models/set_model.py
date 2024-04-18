from torchvision.models import resnet50, resnet34, alexnet, vit_b_16, densenet121, efficientnet_b2
from transformers import Swinv2ForImageClassification
import torch
from torch import nn


def set_model(model_arg: str, num_labels: int, labels: list):

    if model_arg == 'resnet50':
        model = resnet50(weights='IMAGENET1K_V1')
        img_size = int(224) 

        model.fc = nn.Sequential(
            #nn.Dropout(p=0.2),
            nn.Linear(in_features=model.fc.in_features, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, num_labels),
        )
        # TODO ONLY LAST LAYER
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_arg == 'resnet34':
        model = resnet34(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_labels, bias=True)
        img_size = int(224)     

    elif model_arg == 'swin':
        id2label = {id: label for id, label in enumerate(labels)}
        label2id = {label: id for id, label in id2label.items()}

        img_size = int(256) 

        model = Swinv2ForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256", 
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        raise NotImplementedError("Swin Transformer is not yet supported")
    
    elif model_arg == "vit":
        img_size = int(224) 
        model = vit_b_16(weights="IMAGENET1K_V1")

        model.heads = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels),
        )

    elif model_arg == "alexnet": 
        model = alexnet(weights="DEFAULT")
        img_size = int(224) 
    
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256 * 6 * 6, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, num_labels),
        )
    elif model_arg == "densenet121":
        img_size = int(224) 
        model = densenet121(weights="IMAGENET1K_V1")
        
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, num_labels),
        )
    
    elif model_arg == "efficientnet":
        img_size = int(260)
        model = efficientnet_b2(weights="IMAGENET1K_V1")

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1408, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels),
        )

    else:
        raise ValueError(f'Invalid model argument: {model_arg}')
        
    return model, img_size
     