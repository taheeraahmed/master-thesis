import torch
import yacs
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from utils import FileManager, show_batch_images
from models import ModelConfig, set_optimizer, set_scheduler
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
    model = model_config.model
    model_name = model_config.model_arg
    experiment_name = model_config.experiment_name
    criterion = model_config.criterion
    learning_rate = model_config.learning_rate
    num_labels = model_config.num_labels
    labels = model_config.labels
    optimizer_func = set_optimizer(model_config)
    scheduler_func = set_scheduler(model_config, optimizer_func)
    model_ckpts_folder = file_manager.model_ckpts_folder
    logger = file_manager.logger
    root_path = file_manager.root
    checkpoint_path = model_config.checkpoint_path

    num_workers = model_config.num_cores
    pin_memory = False

    if model_config.fast_dev_run:
        file_manager.logger.info('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)
        test_df = test_df.head(val_subset_size)

    train_transforms, val_transforms, test_transforms = set_transforms(
        model_config, file_manager)

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
    tb_logger = TensorBoardLogger(
        save_dir=f'{file_manager.model_ckpts_folder}')

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

    training_module = MultiLabelLightningModule(
        model=model,
        criterion=criterion,
        learning_rate=learning_rate,
        num_labels=num_labels,
        labels=labels,
        optimizer_func=optimizer_func,
        scheduler_func=scheduler_func,
        model_ckpts_folder=model_ckpts_folder,
        file_logger=logger,
        root_path=root_path,
        model_name=model_name,
        experiment_name=experiment_name,
        img_size=model_config.img_size,
    )

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)
        ckpt_state_dict = state_dict['model']
        training_module.load_state_dict(ckpt_state_dict, strict=False)
        file_manager.logger.info(f"🚀 Loaded the model from {checkpoint_path}")
    else:
        file_manager.logger.info('🚀 Training the model from scratch')

    pl_trainer = Trainer(
        max_epochs=model_config.num_epochs,
        logger=tb_logger,
        fast_dev_run=model_config.fast_dev_run,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    if not model_config.eval_mode:
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
