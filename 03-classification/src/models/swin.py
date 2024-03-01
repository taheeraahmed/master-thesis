import torch
from transformers import Swinv2ForImageClassification, Trainer, SwinConfig
from torch import nn
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import sys

from data.chestxray14 import ChestXray14SwinDataset
from utils.df import get_df
from utils.handle_class_imbalance import get_class_weights
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def compute_metrics(p):
    # Convert logits to predicted class indices
    preds = np.argmax(p.predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    
    # For AUC, we need probability scores. The following assumes binary classification
    # and uses the probability of the positive class (class index 1)
    # For multi-class classification, you'll need a different approach
    if p.predictions.shape[1] == 2:
        # Use probabilities of the positive class for AUC calculation
        probs = p.predictions[:, 1]
        auc = roc_auc_score(p.label_ids, probs)
    else:
        # Dummy value in case of multi-class classification; adjust as needed
        auc = float('nan')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
    }

def swin(logger, args, idun_datetime_done, data_path):
    logger.info('Using Swin Transformer model from HF and also using HF Trainer')
    train_df, val_df, labels = get_df(args, data_path, logger)


    num_classes = 14
    model_name = "microsoft/swinv2-tiny-patch4-window8-256"

    if args.test_mode:
        logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = ChestXray14SwinDataset(
        model_name=model_name, dataframe=train_df, transforms=transforms)

    val_dataset = ChestXray14SwinDataset(
        model_name=model_name, dataframe=val_df, transforms=transforms)

    # Making sure the class weights have the correct shape
    class_weights = get_class_weights(train_df)
    assert class_weights.size(0) == 14, 'Class weights should be of size 14'
    
    config = SwinConfig.from_pretrained(model_name, num_labels=14)
    model = Swinv2ForImageClassification.from_pretrained(
        model_name, config=config)
    
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    # TODO: Noe rart med shapes??? skjønner ikke hva som skjer :)) ææ
    if args.loss == 'wce':
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
        trainer = Trainer(
            model=model,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            class_weights=class_weights,         # class weights
            eval_dataset=val_dataset,            # evaluation dataset
            compute_metrics=compute_metrics,     # function to compute metrics
        )
        trainer.train()
        trainer.evaluate()
    elif args.loss == 'wfl':
        raise NotImplementedError
    else:
        logger.error('Invalid loss function argument')
        sys.exit(1)
