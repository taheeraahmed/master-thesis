import torch.nn as nn
import torch
from trainers.class_trainer import TrainerClass
from torch.utils.data import DataLoader
from torchvision import transforms
import torchxrayvision as xrv
from data.chestxray14 import ChestXray14Dataset
from utils.df import get_df


def densenet121(model_config, file_manager):
    shuffle = True
    num_workers = 4

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    
    train_df, val_df, labels = get_df(file_manager)

    model = xrv.models.get_model(weights="densenet121-res224-nih")
    model.op_threshs = None

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, len(labels)),
    )

    # only training classifier
    optimizer = torch.optim.Adam(model.classifier.parameters())

    if model_config.test_mode:
        file_manager.logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_dataset = ChestXray14Dataset(dataframe=train_df, transform=transform, labels=labels)
    train_dataloader = DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=shuffle, num_workers=num_workers)

    val_dataset = ChestXray14Dataset(dataframe=val_df, transform=transform, labels=labels)
    validation_dataloader = DataLoader(
        val_dataset, batch_size=model_config.batch_size, shuffle=shuffle, num_workers=num_workers)

    trainer = TrainerClass(
        model_config = model_config,
        file_manager = file_manager,
        model=model,
        classnames=labels,
        optimizer=optimizer,
    )
    trainer.train(
        model_config=model_config,
        file_manager=file_manager,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
    )
