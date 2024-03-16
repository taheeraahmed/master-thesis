from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
from torchmetrics.classification import Accuracy, F1Score, AUROC
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    Resize,
                                    ToTensor)
import torch.nn.functional as F
import torch
from data.chestxray14 import ChestXray14HFDataset
from utils.df import get_df
from utils import FileManager, ModelConfig, FocalLoss


class ChestXray14Model(LightningModule):
    def __init__(self, num_labels, model_name, criterion, learning_rate=2e-5, class_weights=None):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=num_labels)
        self.learning_rate = learning_rate
        # Initialize metrics
        self.accuracy = Accuracy()
        self.f1_score = F1Score(num_classes=num_labels, average='macro')
        self.auroc = AUROC(num_classes=num_labels,
                           average='macro', compute_on_step=False)

        self.criterion = criterion

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def step(self, batch):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        # for multi-class classification -- using softmax
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # update training metrics for accuracy and F1
        self.accuracy(preds, labels)
        self.f1_score(preds, labels)

        # for AUROC use probabilities, ensure labels formatted
        if self.auroc.num_classes == 2:
            # Binary classification scenario
            self.auroc(probs[:, 1], labels)
        else:
            # multi-class scenario
            one_hot_labels = F.one_hot(
                labels, num_classes=self.model.num_labels)
            self.auroc(probs, one_hot_labels)

        # log metrics
        self.log('train_accuracy', self.accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.f1_score, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.auroc, on_step=False,
                 on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.accuracy(preds, labels)
        self.f1_score(preds, labels)
        if self.auroc.num_classes == 2:
            self.auroc(probs[:, 1], labels)
        else:
            one_hot_labels = F.one_hot(
                labels, num_classes=self.model.num_labels)
            self.auroc(probs, one_hot_labels)

        self.log('val_accuracy', self.accuracy, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1_score, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.auroc, on_epoch=True, prog_bar=True)


def swin(model_config: ModelConfig, file_manager: FileManager) -> None:
    model_name = "microsoft/swin-base-patch4-window12-384"

    train_df, val_df, _, class_weights = get_df(file_manager)

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


    train_dataset = ChestXray14HFDataset(
        dataframe=train_df, model_name=model_name, transform=train_transforms)
    val_dataset = ChestXray14HFDataset(
        dataframe=val_df, model_name=model_name, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=model_config.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=model_config.batch_size, shuffle=False)
    
    # TODO: Change to experiment name in file_manager
    logger = TensorBoardLogger(save_dir=file_manager.model_ckpts_folder, name=file_manager.output_folder)
    if model_config.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    if model_config.loss == 'wce':
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif model_config.loss == 'wfl':
        criterion = FocalLoss(alpha=class_weights)

    model = ChestXray14Model(
        num_labels=model_config.num_classes,
        criterion=criterion,
        model_name=model_name,
        learning_rate=model_config.learning_rate,
        class_weights=class_weights
    )

    trainer = Trainer(max_epochs=model_config.num_epochs,
                      logger=logger,
                      gpus=1, 
                      progress_bar_refresh_rate=20
    )

    trainer.fit(model, 
                train_dataloader=train_loader,
                val_dataloaders=val_loader
    )
