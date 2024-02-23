import torch
from transformers import Swinv2ForImageClassification
from trainers.multiclass_trainer import TrainerClass
from torch.utils.data import DataLoader
from datasets import ChestXray14SwinDataset
from utils.df import get_df_image_paths_labels
from utils.handle_class_imbalance import get_class_weights


def swin(logger, args, idun_datetime_done, data_path):
    shuffle = True
    num_workers = 4

    model = Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256")
    
    model.op_threshs = None
    model.classifier = torch.nn.Linear(768, 14)

    optimizer = torch.optim.Adam(model.classifier.parameters())

    train_df, val_df = get_df_image_paths_labels(args, data_path, logger)
    if args.test_mode:
        logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_dataset = ChestXray14SwinDataset(
        dataframe=train_df, model_name="microsoft/swinv2-tiny-patch4-window8-256")
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

    val_dataset = ChestXray14SwinDataset(dataframe=val_df, model_name="microsoft/swinv2-tiny-patch4-window8-256")
    validation_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

    class_weights = get_class_weights(train_df)

    trainer = TrainerClass(
        model=model,
        model_name = args.model,
        loss = args.loss,
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
