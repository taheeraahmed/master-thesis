from torchvision.models import resnet50, alexnet, vit_b_16, densenet121, efficientnet_b1, swin_b
import timm
from transformers import Swinv2ForImageClassification
import torch
from torch import nn


def classifying_head(in_features: int, num_labels: int):
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=in_features, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(128, num_labels),
    )

def freeze_backbone(model):
    for param in model.parameters():
            param.requires_grad = False

    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def set_model(model_arg: str, num_labels: int, labels: list):

    if model_arg == 'resnet50':
        model = resnet50(weights='IMAGENET1K_V2')
        img_size = int(224)
        input_features = model.fc.in_features

        model.fc = classifying_head(input_features, num_labels)
        #model.fc = nn.Linear(input_features, num_labels, bias=True)

        # step 1: TODO ONLY LAST LAYER
        # for param in model.parameters():
        #     param.requires_grad = True

        # for param in model.fc.parameters():
        #     param.requires_grad = False

    elif model_arg == 'swin':
        img_size = int(224)
        model = timm.create_model(
            'swin_base_patch4_window7_224', num_classes=num_labels)
        
        #model = freeze_backbone(model)

    elif model_arg == "vit":
        # img_size = int(224)
        # model = vit_b_16(weights="IMAGENET1K_V1")

        # model.heads = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(768, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_labels),
        # )
        raise NotImplementedError("Vision Transformer is not yet supported")

    elif model_arg == "alexnet":
        model = alexnet(weights="DEFAULT")
        img_size = int(224)

        input_features = 256*6*6

        model.classifier = classifying_head(input_features, num_labels)
        #model.classifier = nn.Linear(256*6*6, num_labels, bias=True)

    elif model_arg == "densenet121":
        img_size = int(224)
        model = densenet121(weights="IMAGENET1K_V1")

        model.classifier = classifying_head(1024, num_labels)
        #model.classifier = nn.Linear(1024, num_labels, bias=True)

    elif model_arg == "efficientnet":
        img_size = int(260)
        model = efficientnet_b1(weights="IMAGENET1K_V2")

        model.classifier = nn.Linear(1280, num_labels, bias=True)

    else:
        raise ValueError(f'Invalid model argument: {model_arg}')

    return model, img_size
