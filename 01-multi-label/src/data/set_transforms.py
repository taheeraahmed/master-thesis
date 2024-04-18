

from torchvision import transforms
from utils import FileManager
from models import ModelConfig

from torchvision.transforms import functional as F
from torchvision import transforms
class AdjustSharpness:
    def __init__(self, sharpness_factor):
        super().__init__()
        self.sharpness_factor = sharpness_factor
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be gamma adjusted.

        Returns:
            PIL Image or Tensor: Gamma adjusted image.
        """
        return F.adjust_sharpness(img, self.sharpness_factor)
    
class AdjustGamma:
    def __init__(self, gamma, gain=1):
        super().__init__()
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be gamma adjusted.

        Returns:
            PIL Image or Tensor: Gamma adjusted image.
        """
        return F.adjust_gamma(img, self.gamma, self.gain)
    
def set_transforms(model_config: ModelConfig, file_manager: FileManager):
    if model_config.add_transforms:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # https://github.com/thtang/CheXNet-with-localization/blob/master/train.py#L100
        train_transforms = transforms.Compose([
            #transforms.Resize(size=(model_config.img_size, model_config.img_size)),
            #transforms.RandomResizedCrop(model_config.img_size),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(model_config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(model_config.img_size),
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

    