from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np


class ChestXray14HFDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            model_name (str): The name of the Swin Transformer model you're using.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')

        labels = self.dataframe.iloc[idx, 1:].to_numpy(dtype='float32')
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": labels}


class ChestXray14Dataset(Dataset):
    def __init__(self, data_path, file_path, augment, num_class=14):
        self.augment = augment
        columns = ['Image Filename'] + [f'label{i}' for i in range(num_class)]
        
        # Read the data file into a DataFrame
        df = pd.read_csv(file_path, sep='\s+', names=columns)
        
        # Mapping image filenames to full paths
        subfolders = [f"images_{i:03}/images" for i in range(1, 13)]
        path_mapping = {}
        for subfolder in subfolders:
            full_folder_path = os.path.join(data_path, subfolder)
            if os.path.isdir(full_folder_path):  # Ensure directory exists
                for img_file in os.listdir(full_folder_path):
                    path_mapping[img_file] = os.path.join(full_folder_path, img_file)
        
        df['Full Image Path'] = df['Image Filename'].map(path_mapping)
        df = df.dropna(subset=['Full Image Path'])  # Drop entries without a valid path


        # Extract labels
        self.img_list = df['Full Image Path'].tolist()
        self.img_label = df[[f'label{i}' for i in range(num_class)]].astype(np.float32).values  # Correct slicing for labels

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.from_numpy(self.img_label[index])
        
        if self.augment:
            imageData = self.augment(imageData)

        return {"pixel_values": imageData, "labels": imageLabel}

    def __len__(self):
        return len(self.img_list)