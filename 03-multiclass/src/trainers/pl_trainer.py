from pytorch_lightning import LightningModule
from transformers import AutoModelForImageClassification
from torchmetrics.classification import Accuracy, F1Score, AUROC
import torch.nn.functional as F
import torch

class TrainerPL(LightningModule):
    def __init__(self, num_labels, model_name, labels, criterion, learning_rate=2e-5):
        super().__init__()

        id2label = {id: label for id, label in enumerate(labels)}
        label2id = {label: id for id, label in id2label.items()}

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=14,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True  # Ignore the size mismatch in the final layer
        )

        self.learning_rate = learning_rate
        # initialize metrics for multi-class classification
        self.accuracy = Accuracy(
            task='multiclass',
            num_classes=num_labels,
            average='macro',
        )
        self.f1_score = F1Score(
            task='multiclass',
            num_classes=num_labels,
            average='macro'
        )
        self.auroc = AUROC(
            task='multiclass',
            num_classes=num_labels,
            average='macro',
            compute_on_step=False
        )

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
