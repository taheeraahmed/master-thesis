from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


class ChestXray14SwinDataset(Dataset):
    def __init__(self, dataframe, model_name):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            model_name (str): The name of the Swin Transformer model you're using.
        """
        self.dataframe = dataframe
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.labels = self._get_labels()

    def _get_labels(self):
        labels = self.dataframe['Finding Labels'].str.split(
            '|').explode().unique()
        labels.sort()
        return labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')

        # Process image
        processed_image = self.processor(
            images=image, return_tensors="pt").pixel_values[0]

        labels = self.dataframe.iloc[idx, 1:].to_numpy(dtype='float32')
        labels = torch.tensor(labels)

        return {"pixel_values": processed_image, "labels": labels}


class ChestXray14Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        ChestXray14 Dataset using DenseNet121 from torchxrayvision pre-trained on the ChestXray14 dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.labels = self._get_labels()

        # TODO find underrepresented diseases

    def _get_labels(self):
        labels = self.dataframe['Finding Labels'].str.split(
            '|').explode().unique()
        labels.sort()
        return labels

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
