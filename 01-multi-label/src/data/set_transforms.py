from torchvision import transforms
from utils import FileManager
from models import ModelConfig

from torchvision import transforms
import torch


def set_test_transforms(logger, img_size, normalize):
    test_transforms = transforms.Compose([
        transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.TenCrop(img_size),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack(
            [normalize(crop) for crop in crops])),
    ])
    logger.info(f"Test time augmentations: \n {test_transforms}")
    return test_transforms


def set_transforms(model_config: ModelConfig, file_manager: FileManager):
    img_size = model_config.img_size
    # imagenet
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # cxr
    normalize = transforms.Normalize(
        [0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    if model_config.add_transforms:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(
                (img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

        if model_config.test_time_augmentation:
            file_manager.logger.info(
                f"Train augmentations: \n {train_transforms}")
            file_manager.logger.info(
                f"Validation augmentations: \n {val_transforms}")
            test_transforms = set_test_transforms(
                file_manager.logger, img_size, normalize)
            return train_transforms, val_transforms, test_transforms

        file_manager.logger.info(f"Train augmentations: \n {train_transforms}")
        file_manager.logger.info(
            f"Validation/Test augmentations: \n {val_transforms}")
        return train_transforms, val_transforms, val_transforms

    elif not model_config.add_transforms:
        train_transforms = transforms.Compose([
            transforms.Resize(
                (img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(
                (img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
        ])

        if model_config.test_time_augmentation:
            file_manager.logger.info(
                f"Train augmentations: \n {train_transforms}")
            file_manager.logger.info(
                f"Validation augmentations: \n {val_transforms}")
            test_transforms = set_test_transforms(
                file_manager.logger, img_size, normalize)
            return train_transforms, val_transforms, test_transforms

        file_manager.logger.info(f"Train augmentations: \n {train_transforms}")
        file_manager.logger.info(
            f"Validation/Test augmentations: \n {val_transforms}")
        return train_transforms, val_transforms, val_transforms


def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence