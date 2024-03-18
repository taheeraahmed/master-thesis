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
from data.chestxray14 import ChestXray14Dataset
from utils.df import get_df
from utils import FileManager, ModelConfig
import torchxrayvision as xrv
from trainers import MultiLabelModelTrainer


def densenet121(model_config: ModelConfig, file_manager: FileManager) -> None:
    model = xrv.models.get_model(weights="densenet121-res224-nih")

    train_df, val_df, labels, class_weights = get_df(file_manager, one_hot=False)

    if model_config.test_mode:
        file_manager.logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_transforms = Compose([
        Resize(256),
        CenterCrop(256),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = Compose([
        Resize(256),
        CenterCrop(256),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ChestXray14Dataset(
        dataframe=train_df, transform=train_transforms)
    val_dataset = ChestXray14Dataset(
        dataframe=val_df, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=model_config.batch_size, shuffle=False)

    logger = TensorBoardLogger(
        save_dir=file_manager.output_folder, name=file_manager.output_folder)
    
    if model_config.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif model_config.loss == 'wce':
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif model_config.loss == 'wfl':
        criterion = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=class_weights,
            gamma=2,
            reduction='mean',
            force_reload=False
        )

    training_module = MultiLabelModelTrainer(
        file_manager=file_manager,
        num_labels=len(labels),
        criterion=criterion,
        labels=labels,
        model=model,
        learning_rate=model_config.learning_rate,
    )

    pl_trainer = Trainer(
        max_epochs=model_config.num_epochs,
        logger=logger,
        gpus=1,
        fast_dev_run=model_config.test_mode,
        max_steps=10 if model_config.test_mode else model_config.max_steps
    )

    pl_trainer.fit(training_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
                )
