from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy, F1Score, AUROC
import torch
import torch.nn.functional as F

class MultiLabelModelTrainer(LightningModule):
    def __init__(self, file_manager, model, num_labels, labels, criterion, learning_rate=2e-5):
        super().__init__()
        self.file_manager = file_manager
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.num_labels = num_labels

        # Adjust metrics for multi-label
        self.accuracy = Accuracy(num_labels=num_labels, average='macro', task='multilabel')
        self.f1_score = F1Score(num_labels=num_labels, average='macro', task='multilabel')
        # Uncomment and adjust AUROC for multi-label if needed
        # self.auroc = AUROC(num_classes=num_labels, average='macro', compute_on_step=False, task='multilabel')

    def forward(self, pixel_values):
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        probs = torch.sigmoid(logits)

        # Update metrics
        self.accuracy(probs, labels)
        self.f1_score(probs, labels)

        # Log metrics
        self.log('train_accuracy', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.f1_score, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = torch.sigmoid(logits)

        # Update metrics
        self.accuracy(probs, labels)
        self.f1_score(probs, labels)

        # Log metrics
        self.log('val_accuracy', self.accuracy, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1_score, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
        #scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 'monitor': 'val_loss'}
        
        #return [optimizer], [scheduler]

