from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torchvision import transforms   

from utils import FileManager
from models import ModelConfig
from trainers import MultiLabelLightningModule
from data import ChestXray14HFDataset

import copy
import random
import numbers
from collections.abc import Sequence
import warnings
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import ImageFilter


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

def train_and_evaluate_model(model_config: ModelConfig, file_manager: FileManager, train_df, val_df, test_df) -> None:
    """
    Start the training and validation of the model
    :param model_config: ModelConfig object
    :param file_manager: FileManager object
    :param train_df: DataFrame containing the training data
    :param val_df: DataFrame containing the validation data
    """
    
    num_workers = 4
    pin_memory = False

    if model_config.test_mode:
        file_manager.logger.info('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_transforms = transforms.Compose([
        transforms.Resize((model_config.img_size, model_config.img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
        transforms.RandomRotation(degrees=(0, 45)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    file_manager.logger.info(f"Using these augmentations: {train_transforms}")

    val_transforms = transforms.Compose([
        transforms.Resize((model_config.img_size, model_config.img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ChestXray14HFDataset(
        dataframe=train_df, transform=train_transforms)
    val_dataset = ChestXray14HFDataset(
        dataframe=val_df, transform=val_transforms)
    test_dataset = ChestXray14HFDataset(
        dataframe=test_df, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    logger = TensorBoardLogger(save_dir=f'{file_manager.model_ckpts_folder}')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1, 
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',  
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )


    training_module = MultiLabelLightningModule(
        model_config=model_config,
        file_manager=file_manager,
    )

    pl_trainer = Trainer(
        max_epochs=model_config.num_epochs,
        logger=logger,
        fast_dev_run=model_config.test_mode,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    pl_trainer.fit(
        training_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if not model_config.test_mode:
        pl_trainer.test(
            model=training_module,
            dataloaders=test_loader,
        )