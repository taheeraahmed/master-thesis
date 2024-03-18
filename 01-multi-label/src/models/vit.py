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
from utils import FileManager, ModelConfig, set_criterion
from trainers import MulticlassModelTrainer
from transformers import AutoModelForImageClassification


def vit(model_config: ModelConfig, file_manager: FileManager) -> None:
    model_name = "google/vit-base-patch16-224-in21k"

    train_df, val_df, labels, class_weights = get_df(file_manager, one_hot=False)

    if model_config.test_mode:
        file_manager.logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_transforms = Compose([
        Resize(224),
        CenterCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = Compose([
        Resize(224),
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
        save_dir=file_manager.output_folder, name=file_manager.output_folder)
    
    criterion = set_criterion(model_config.loss, class_weights)

    id2label = {id: label for id, label in enumerate(labels)}
    label2id = {label: id for id, label in id2label.items()}

    model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

    training_module = MulticlassModelTrainer(
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
        max_steps=10 if model_config.test_mode else None
    )

    pl_trainer.fit(training_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
                )
