from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
import timm
import random
import torch.nn as nn
from memory_profiler import memory_usage
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import time
import psutil

print("started script")

root_folder = '/cluster/home/taheeraa/datasets/chestxray-14/'
images_path = f"{root_folder}images"
file_path_bbox = root_folder + 'BBox_List_2017.csv'

labels = [
        "Atelectasis", 
        "Cardiomegaly",
        "Effusion", 
        "Infiltration", 
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",  
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia"
    ]
num_labels = len(labels) 

base = "/cluster/home/taheeraa/code/BenchmarkTransformers/models/classification/ChestXray14/"

model_str = 'vit_in1k'

if model_str == "densenet121":
  experiment_name = '05-change-classifier-head'
  model_type = 'densenet121_imagenet_1k_adamw_32_bce_aug_class'
elif model_str == "swin_simim":
  experiment_name = "07-transformer-ssl/"
  model_type = "swin_base_simmim"
elif model_str == "swin_in22k":
  experiment_name = '06-transformers-pre-trained'
  model_type = 'swin_base_imagenet_1k_sgd_64_bce_aug'
elif model_str == "vit_in1k":
  experiment_name = '06-transformers-pre-trained'
  model_type = 'vit_base_imagenet_1k_sgd_64_bce_True'

pretrained_weights = os.path.join(base, experiment_name, model_type, "model.pth.tar")
batch_size = 32

print(model_str, batch_size, pretrained_weights)

if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")

print(device)

def classifying_head(in_features: int, num_labels: int):
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=in_features, out_features=128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(128, num_labels),
    )

def load_model(pretrained_weights, num_labels, model_str):
    
    checkpoint = torch.load(
        pretrained_weights, map_location=torch.device('cpu'))
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
    else: normalization = "imagenet"

    checkpoint = torch.load(pretrained_weights, map_location="cpu")

    state_dict = checkpoint['state_dict']
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded with msg: {}'.format(msg))
    return model, normalization

model, normalization = load_model(pretrained_weights, num_labels, model_str)

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

class ChestXray14Dataset(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=14, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return {'pixel_values': imageData, 'labels': imageLabel}

  def __len__(self):

    return len(self.img_list)
  

test_transforms = build_transform_classification(
        mode = "test",
        normalize=normalization,
        test_augment=True,
    )

path_to_labels = '/cluster/home/taheeraa/code/BenchmarkTransformers/dataset'
file_path_train = path_to_labels + '/Xray14_train_official.txt'
file_path_val = path_to_labels + '/Xray14_val_official.txt'
file_path_test = path_to_labels + '/Xray14_test_official.txt'

test_dataset = ChestXray14Dataset(images_path=images_path, file_path=file_path_test,
                                      augment=test_transforms, num_class=num_labels)


test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

model.to(device)
model.eval()
model = model.to(device)

y_test = torch.FloatTensor().to(device)
p_test = torch.FloatTensor().to(device)

memory_usage = []
inference_times = []
process = psutil.Process()

# Limit to processing only 10 samples
sample_count = 0
max_samples = 10

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        if sample_count >= max_samples:
            break
        samples = batch['pixel_values']
        targets = batch['labels']
        
        targets = targets.to(device)
        y_test = torch.cat((y_test, targets), 0)

        if len(samples.size()) == 4:
            bs, c, h, w = samples.size()
            n_crops = 1
        elif len(samples.size()) == 5:
            bs, n_crops, c, h, w = samples.size()

        varInput = torch.autograd.Variable(
            samples.view(-1, c, h, w).to(device))

        # Measure inference time
        start_time = time.time()
        out = model(varInput)
        out = torch.sigmoid(out)
        inference_time = time.time() - start_time

        outMean = out.view(bs, n_crops, -1).mean(1)
        p_test = torch.cat((p_test, outMean.data), 0)

        # Record inference time
        inference_times.append(inference_time)

        # Measure memory usage
        memory_info = process.memory_info()
        memory_usage.append(memory_info.rss)
        sample_count += 1

average_inference_time = sum(inference_times) / len(inference_times)
average_memory_usage = sum(memory_usage) / len(memory_usage)

print(f'Average inference time: {average_inference_time:.6f} seconds')
print(f'Average memory usage: {average_memory_usage / (1024 ** 2):.2f} MB')

results = pd.DataFrame({
    'Inference Time (s)': inference_times,
    'Memory Usage (bytes)': memory_usage
})

results.to_csv(f'{model_str}_inference_memory_usage.csv', index=False)

