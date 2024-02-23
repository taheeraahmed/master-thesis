import torch.nn as nn
import torch
from trainers.multiclass_trainer import TrainerClass
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchxrayvision as xrv
from data.chestxray14 import ChestXray14Dataset
from utils.df import get_df_image_paths_labels
from utils.handle_class_imbalance import get_class_weights


def densenet121(logger, args, idun_datetime_done, data_path):
    shuffle = True
    num_workers = 4

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    model = xrv.models.get_model(weights="densenet121-res224-nih")
    model.op_threshs = None

    # Change the last layer to output 14 classes and use sigmoid activation for mulitlabel classification
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
        nn.Sigmoid()
    )

    # only training classifier
    optimizer = torch.optim.Adam(model.classifier.parameters())

    train_df, val_df = get_df_image_paths_labels(args, data_path, logger)
    if args.test_mode:
        logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_dataset = ChestXray14Dataset(dataframe=train_df, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

    val_dataset = ChestXray14Dataset(dataframe=val_df, transform=transform)
    validation_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

    class_weights = get_class_weights(train_df)

    trainer = TrainerClass(
        model=model,
        model_name=args.model,
        loss=args.loss,
        class_weights=class_weights,
        model_output_folder=f'output/{args.output_folder}/model_checkpoints',
        logger=logger,
        log_dir=f'output/{args.output_folder}',
        optimizer=optimizer,
    )
    trainer.train(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        num_epochs=args.num_epochs,
        idun_datetime_done=idun_datetime_done,
        model_arg=args.model
    )
