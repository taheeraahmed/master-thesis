import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from utils import FileManager, show_batch_images
from models import ModelConfig
from trainers import MultiLabelLightningModule
from data import ChestXray14HFDataset, set_transforms


def train_and_evaluate_model(model_config: ModelConfig, file_manager: FileManager, train_df, val_df, test_df) -> None:
    """
    Start the training and validation of the model
    :param model_config: ModelConfig object
    :param file_manager: FileManager object
    :param train_df: DataFrame containing the training data
    :param val_df: DataFrame containing the validation data
    """
    
    num_workers = model_config.num_cores
    pin_memory = False

    if model_config.fast_dev_run:
        file_manager.logger.info('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)
        test_df = test_df.head(val_subset_size)

    train_transforms, val_transforms, test_transforms = set_transforms(model_config, file_manager)

    train_dataset = ChestXray14HFDataset(
        dataframe=train_df, transform=train_transforms)
    val_dataset = ChestXray14HFDataset(
        dataframe=val_df, transform=val_transforms)
    test_dataset = ChestXray14HFDataset(
        dataframe=test_df, transform=test_transforms)

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

    show_batch_images(file_manager=file_manager, dataloader=train_loader)
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
        patience=5,
        verbose=True,
        mode='min'
    )

    checkpoint_path = None
    #checkpoint_path = "/cluster/home/taheeraa/code/master-thesis/01-multi-label/output/v0-experiments/013-train-in-two-steps/2024-04-19-16:24:29-resnet50-bce-14-multi-label-e35-bs128-lr0.0005-step-two-train-backbone/model_checkpoints/lightning_logs/version_0/checkpoints/epoch=7-step=4712.ckpt"
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        adjusted_state_dict = {key.replace('model.model.', 'model.'): value 
                            for key, value in checkpoint['state_dict'].items()}
        training_module = MultiLabelLightningModule(model_config=model_config, file_manager=file_manager)
        training_module.load_state_dict(adjusted_state_dict, strict=False)
        file_manager.logger.info(f"🚀 Loaded the model from {checkpoint_path}")
    else:
        file_manager.logger.info('🚀 Training the model from scratch')
        training_module = MultiLabelLightningModule(
            model_config=model_config,
            file_manager=file_manager,
        )
        
    pl_trainer = Trainer(
        max_epochs=model_config.num_epochs,
        logger=logger,
        fast_dev_run=model_config.fast_dev_run,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    if not model_config.test_mode:
        pl_trainer.fit(
            training_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    file_manager.logger.info('✅ Training is done')

    pl_trainer.test(
        model=training_module,
        dataloaders=test_loader,
    )

    file_manager.logger.info('✅ Testing is done')