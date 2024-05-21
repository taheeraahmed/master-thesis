from torchvision import transforms
from utils import FileManager
from models import ModelConfig

from torchvision import transforms
import torch


def build_transform_classification(normalize, test_augment, add_transforms, crop_size=224, resize=256):
    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)

    if add_transforms:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor(),
            normalize
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])    

        if test_augment:
            test_transforms = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.TenCrop(crop_size),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])
        else:
            test_transforms = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize
            ])

    elif not add_transforms:
        if test_augment:
           raise NotImplementedError("Test-time augmentation is not supported without additional transforms")
        if normalize is None:
            train_transforms = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])

            val_transforms = train_transforms
            test_transforms = train_transforms
        else:
            train_transforms = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize
            ])
            val_transforms = train_transforms
            test_transforms = train_transforms
    
    return train_transforms, val_transforms, test_transforms
