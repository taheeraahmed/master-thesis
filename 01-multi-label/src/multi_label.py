from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    Resize,
                                    ToTensor)
import torch
from utils import FileManager
from models import ModelConfig
from trainers import MultiLabelLightningModule
from data import ChestXray14HFDataset
from torchvision.transforms import InterpolationMode    


def train_and_evaluate_model(model_config: ModelConfig, file_manager: FileManager, train_df, val_df, test_df, labels) -> None:
    num_workers = 4
    pin_memory = False
    if model_config.test_mode:
        file_manager.logger.info('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_transforms = Compose([
        Resize((model_config.img_size, model_config.img_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #RandomHorizontalFlip(),
    ])

    val_transforms = Compose([
        Resize((model_config.img_size, model_config.img_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        monitor='val_loss',  # Monitor F1 score validation metric
        mode='min',
        save_top_k=1,  # Save the best checkpoint only
        verbose=True,  # Print a message whenever a new checkpoint is saved
    )

    training_module = MultiLabelLightningModule(
        model_config=model_config,
        file_manager=file_manager,
    )

    pl_trainer = Trainer(
        max_epochs=model_config.num_epochs,
        logger=logger,
        # gpus=1,
        fast_dev_run=model_config.test_mode,
        #max_steps=10 if model_config.test_mode else model_config.max_steps,
        callbacks=[checkpoint_callback],
    )

    pl_trainer.fit(
        training_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if model_config.test_mode == False:
        pl_trainer.test(
            dataloaders=test_loader,
            ckpt_path='best',
        )