
import torch
import torchvision.transforms as transforms
import PIL.Image as Image


def build_transform_classification(normalize, crop_size=224, resize=256, tta=True):
    transformations_list = []
    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize(
          [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize(
          [0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if tta:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformations_list.append(transforms.Lambda(
            lambda crops: torch.stack([normalize(crop) for crop in crops])))
    else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())

    transformSequence = transforms.Compose(transformations_list)
    print(transformSequence)
    return transformSequence
