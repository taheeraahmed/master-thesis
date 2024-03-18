from data.chestxray14 import ChestXray14HFDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from transformers import TrainingArguments
from utils.df import get_df
import torch
from math import ceil
from transformers import Trainer
from transformers import ViTForImageClassification
from torch.nn import CrossEntropyLoss
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    Resize,
                                    ToTensor)
import sys
import torch
from utils import FileManager, ModelConfig


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"]
                               for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


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
        labels = labels.argmax(dim=-1)
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
    threshold = 0.5
    preds_binary = (preds > threshold).astype(int)

    precision = precision_score(labels, preds_binary, average='micro')
    recall = recall_score(labels, preds_binary, average='micro')
    f1 = f1_score(labels, preds_binary, average='micro')

    auc_scores = roc_auc_score(labels, preds, average=None)
    mean_auc = auc_scores.mean()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_auc': mean_auc,
        'auc_scores': auc_scores.tolist(),
    }


def vit(model_config: ModelConfig, file_manager: FileManager) -> None:
    train_df, val_df, labels, class_weights = get_df(file_manager)

    model_name = "microsoft/swinv2-tiny-patch4-window8-256"

    if model_config.test_mode:
        file_manager.logger.warning('Using smaller dataset')
        train_subset_size = 100
        val_subset_size = 50

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_transforms = Compose([
        Resize(256),
        CenterCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ChestXray14HFDataset(
        model_name=model_name, dataframe=train_df, transform=train_transforms)

    val_ds = ChestXray14HFDataset(
        model_name=model_name, dataframe=val_df, transform=val_transforms)

    id2label = {id: label for id, label in enumerate(labels)}
    label2id = {label: id for id, label in id2label.items()}

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                            num_labels=14,
                                                            id2label=id2label,
                                                            label2id=label2id)

    total_steps = int(ceil((train_ds.__len__() / model_config.batch_size) * \
        model_config.num_epochs))

    file_manager.logger.info(f'Total steps: {total_steps}')

    training_args = TrainingArguments(
        output_dir=f'{file_manager.model_ckpts_folder}',
        num_train_epochs=model_config.num_epochs,  # number of training epochs
        max_steps=total_steps,                     # max number of training steps
        # batch size per device during training
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        # directory for storing logs
        logging_dir=f'{file_manager.model_ckpts_folder}',
        logging_steps=10,                # log & save weights each logging_steps
        evaluation_strategy="epoch",     # evaluate each `epoch`
        save_strategy="epoch",           # save checkpoint every epoch
        load_best_model_at_end=True,     # load the best model when finished training
        metric_for_best_model="f1",      # use accuracy to evaluate the best model
        report_to="tensorboard",         # enable logging to TensorBoard
    )

    if model_config.loss == 'wce':
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            class_weights=class_weights,  # Now you can pass class_weights
        )
    elif model_config.loss == 'ce':
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
    elif model_config.loss == 'wfl':
        raise NotImplementedError
    else:
        file_manager.logger.error('Invalid loss function argument')
        sys.exit(1)

    trainer.train()
    trainer.evaluate()
