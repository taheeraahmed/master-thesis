import torch
from transformers import Swinv2ForImageClassification, Trainer, Swinv2Config
from torch.nn import CrossEntropyLoss
import sys

from data.chestxray14 import ChestXray14HFDataset
from utils.df import get_df
from transformers import TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert class weights to a tensor if provided
        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights, dtype=torch.float).to(self.model.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # Compute custom loss
        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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


def swin(model_config, file_manager):
    train_df, val_df, _, class_weights = get_df(file_manager)

    model_name = "microsoft/swinv2-tiny-patch4-window8-256"

    if model_config.test_mode:
        file_manager.logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_dataset = ChestXray14HFDataset(
        model_name=model_name, dataframe=train_df)

    val_dataset = ChestXray14HFDataset(
        model_name=model_name, dataframe=val_df)

    configuration = Swinv2Config()

    model = Swinv2ForImageClassification(configuration)

    model.num_labels = model_config.num_classes

    training_args = TrainingArguments(
        output_dir=f'output/{file_manager.output_folder}',
        num_train_epochs=model_config.num_epochs,  # number of training epochs
        # batch size per device during training
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        # directory for storing logs
        logging_dir=f'output/{file_manager.output_folder}',
        logging_steps=10,                # log & save weights each logging_steps
        evaluation_strategy="epoch",     # evaluate each `epoch`
        save_strategy="epoch",           # save checkpoint every epoch
        load_best_model_at_end=True,     # load the best model when finished training
        metric_for_best_model="f1",  # use accuracy to evaluate the best model
        report_to="tensorboard",         # enable logging to TensorBoard
    )

    if model_config.loss == 'wce':
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights,  # Now you can pass class_weights
        )
    elif model_config.loss == 'ce':
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
    elif model_config.loss == 'wfl':
        raise NotImplementedError
    else:
        file_manager.logger.error('Invalid loss function argument')
        sys.exit(1)

    trainer.train()
    trainer.evaluate()
