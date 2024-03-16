from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


class ChestXray14SwinDataset(Dataset):
    def __init__(self, dataframe, model_name, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            model_name (str): The name of the Swin Transformer model you're using.
        """
        self.dataframe = dataframe
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')

        # Process image
        # image = self.processor(
        #     images=image, return_tensors="pt")

        labels = self.dataframe.iloc[idx, 1:].to_numpy(dtype='float32')
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": labels}


class ChestXray14Dataset(Dataset):
    def __init__(self, dataframe, transform=None, labels=None):
        """
        ChestXray14 Dataset using DenseNet121 from torchxrayvision pre-trained on the ChestXray14 dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx, 0]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"Error opening image file: {img_path}")
            return None

        labels = self.dataframe.iloc[idx, 1:].to_numpy()
        labels = torch.from_numpy(labels.astype('float32'))

        if self.transform:
            image = self.transform(image)

        return {"img": image, "lab": labels}
