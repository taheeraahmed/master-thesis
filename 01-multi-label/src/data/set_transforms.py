

from torchvision import transforms
from utils import FileManager
from models import ModelConfig

from torchvision import transforms
    
def set_transforms(model_config: ModelConfig, file_manager: FileManager):
    img_size = model_config.img_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if model_config.add_transforms:
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(img_size),
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
    
    file_manager.logger.info(f"Train augmentations: \n {train_transforms}")
    file_manager.logger.info(f"Validation augmentations: \n {val_transforms}")
    return train_transforms, val_transforms

    