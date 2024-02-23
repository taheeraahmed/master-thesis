import torch
from transformers import Swinv2ForImageClassification
from torch import nn
from trainers.multiclass_trainer import TrainerHF
import sys

from data.chestxray14 import ChestXray14SwinDataset
from utils.df import get_df_image_paths_labels
from utils.handle_class_imbalance import get_class_weights
from transformers import TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    # Apply threshold to predictions (e.g., 0.5) to convert to binary format if necessary
    threshold = 0.5
    preds_binary = (preds > threshold).astype(int)

    # Ensure you're using metrics that support multilabel-indicator format
    precision = precision_score(labels, preds_binary, average='micro')
    recall = recall_score(labels, preds_binary, average='micro')
    f1 = f1_score(labels, preds_binary, average='micro')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def swin(logger, args, idun_datetime_done, data_path):
    logger.info('Using Swin Transformer model from HF and also using HF Trainer')
    train_df, val_df = get_df_image_paths_labels(args, data_path, logger)
    if args.test_mode:
        logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_dataset = ChestXray14SwinDataset(
        dataframe=train_df, model_name="microsoft/swinv2-tiny-patch4-window8-256")

    val_dataset = ChestXray14SwinDataset(
        dataframe=val_df, model_name="microsoft/swinv2-tiny-patch4-window8-256")

    # Making sure the class weights have the correct shape
    class_weights = get_class_weights(train_df)
    assert class_weights.size(0) == 14, 'Class weights should be of size 14'

    model = Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256")
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
        nn.Sigmoid()
    )

    if args.loss == 'bce':
        training_args = TrainingArguments(
            output_dir=f'output/{args.output_folder}',
            num_train_epochs=args.num_epochs,  # number of training epochs
            # batch size per device during training
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            # directory for storing logs
            logging_dir=f'output/{args.output_folder}',
            logging_steps=10,                # log & save weights each logging_steps
            evaluation_strategy="epoch",     # evaluate each `epoch`
            save_strategy="epoch",           # save checkpoint every epoch
            load_best_model_at_end=True,     # load the best model when finished training
            metric_for_best_model="f1",  # use accuracy to evaluate the best model
            report_to="tensorboard",         # enable logging to TensorBoard
        )
        trainer = TrainerHF(
            model=model,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            class_weights=class_weights,         # class weights
            eval_dataset=val_dataset,            # evaluation dataset
            compute_metrics=compute_metrics,     # function to compute metrics
        )
        trainer.train()
        trainer.evaluate()
    elif args.loss_function == 'focal-loss':
        raise NotImplementedError
    else:
        logger.error('Invalid loss function argument')
        sys.exit(1)
