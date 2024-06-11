import torch
import timm
import torch.nn as nn
from torchvision.models import alexnet


def classifying_head(in_features: int, num_labels: int):
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=in_features, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(128, num_labels),
    )


def load_model(ckpt_path, num_labels, model_str):

    checkpoint = torch.load(
        ckpt_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    if model_str == "densenet121":
        model = timm.create_model(
            'densenet121', num_classes=num_labels, pretrained=True)
        model.classifier = classifying_head(1024, num_labels)
    elif model_str == "swin_simim" or model_str == "swin_in22k":
        model = timm.create_model(
            'swin_base_patch4_window7_224_in22k', num_classes=num_labels, pretrained=True)
    elif model_str == "vit_in1k":
        model = timm.create_model('vit_base_patch16_224',
                                  num_classes=num_labels, pretrained=True)

    if model_str == "swin_simim":
        normalization = "chestx-ray"
    else:
        normalization = "imagenet"

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = checkpoint['state_dict']
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded {model_str} with msg: {msg}')

    img_size = 224
    
    if model is None or normalization is None or img_size is None:
        raise ValueError("Model, normalization, or image size not found")
    
    return model, normalization, img_size
