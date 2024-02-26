import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ChestXray14SwinDataset(Dataset):
    def __init__(self, dataframe, model_name):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            model_name (str): The name of the Swin Transformer model you're using.
        """
        self.dataframe = dataframe
        self.model_name = model_name
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                       'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                       'Pneumonia', 'Pneumothorax']

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')

        # Process image
        processed_image = self.transform(image)
        
        labels = self.dataframe.iloc[idx, 1:].to_numpy(dtype='float32')
        labels = torch.tensor(labels)

        return {"img": processed_image, "lab": labels}


class ChestXray14Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                       'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                       'Pneumonia', 'Pneumothorax']

        underrepresented_diseases = ['Hernia', 'Pneumonia', 'Fibrosis', 'Emphysema',
                                     'Cardiomegaly', 'Pleural_Thickening', 'Consolidation', 'Pneumothorax', 'Mass', 'Nodule']

        self.data_augmentation = transforms.Compose([
            transforms.RandomRotation(0.1),
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])

        self.underrepresented_diseases_indices = [self.labels.index(
            disease) for disease in underrepresented_diseases]

    def _augment_images(self, image, labels):
        ''' 
        Augments the images of the underrepresented diseases.
        '''
        for i in self.underrepresented_diseases_indices:
            if labels[i] == 1:
                return self.data_augmentation(image), labels
        return image, labels

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

        image, labels = self._augment_images(image, labels)

        if self.transform:
            image = self.transform(image)

        return {"img": image, "lab": labels}