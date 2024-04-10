

from torchvision import transforms
from utils import FileManager
from models import ModelConfig
from PIL import ImageFilter
import random
import numbers
from collections.abc import Sequence

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
    def _setup_size(size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size

def set_transforms(model_config: ModelConfig, file_manager: FileManager):
    if model_config.add_transforms:
        normalize = transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252])
        # https://github.com/thtang/CheXNet-with-localization/blob/master/train.py#L100
        train_transforms = transforms.Compose([
            transforms.RandomCrop(model_config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((model_config.img_size, model_config.img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            normalize,
        ])
    
    else:
        file_manager.logger.info("No augmentations are used")
        train_transforms = transforms.Compose([
            transforms.Resize((model_config.img_size, model_config.img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((model_config.img_size, model_config.img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
        ])
    
    file_manager.logger.info(f"Using these augmentations: {train_transforms}")
    return train_transforms, val_transforms

    