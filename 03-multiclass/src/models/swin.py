from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    Resize,
                                    ToTensor)
import torch
from data.chestxray14 import ChestXray14HFDataset
from utils.df import get_df
from utils import FileManager, ModelConfig, FocalLoss
from trainers import TrainerPL


def swin(model_config: ModelConfig, file_manager: FileManager) -> None:
    model_name = "microsoft/swinv2-tiny-patch4-window8-256"

    train_df, val_df, labels, class_weights = get_df(file_manager)

    if model_config.test_mode:
        file_manager.logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_transforms = Compose([
        Resize(256),
        CenterCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ChestXray14HFDataset(
        dataframe=train_df, model_name=model_name, transform=train_transforms)
    val_dataset = ChestXray14HFDataset(
        dataframe=val_df, model_name=model_name, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=model_config.batch_size, shuffle=False)

    logger = TensorBoardLogger(
        save_dir=file_manager.model_ckpts_folder, name=file_manager.output_folder)
    if model_config.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    if model_config.loss == 'wce':
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif model_config.loss == 'wfl':
        criterion = FocalLoss(alpha=class_weights)

    model = TrainerPL(
        num_labels=model_config.num_classes,
        criterion=criterion,
        labels=labels,
        model_name=model_name,
        learning_rate=model_config.learning_rate,
    )

    trainer = Trainer(max_epochs=model_config.num_epochs,
                      logger=logger,
                      gpus=1,
                      progress_bar_refresh_rate=20
                      )

    trainer.fit(model,
                train_dataloader=train_loader,
                val_dataloaders=val_loader
                )
