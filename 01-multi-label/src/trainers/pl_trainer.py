from pytorch_lightning import LightningModule

from torchmetrics.classification import Accuracy, F1Score, AUROC
import torch.nn.functional as F
import torch

class MultiLabelModelTrainer(LightningModule):
    def __init__(self, file_manager, model, num_labels, labels, criterion, learning_rate=2e-5):
        super().__init__()

        
        self.criterion = criterion
        self.file_manager = file_manager

        self.model = model
        self.learning_rate = learning_rate
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
        """ self.auroc = AUROC(
            task='multiclass',
            num_classes=num_labels,
            average='macro',
            compute_on_step=False
        ) """

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def step(self, batch):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        # TODO: This will be incorrect for multilabel
        labels = labels.squeeze(1).long()
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        # for multi-class classification -- using softmax
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.accuracy(preds, labels)
        self.f1_score(preds, labels)
        self.file_manager.logger.info(f"Shapes - probs: {probs.shape}, labels: {labels.shape}")
        #self.auroc(probs, labels)
    
        # log metrics
        self.log('train_accuracy', self.accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.f1_score, on_step=False,
                 on_epoch=True, prog_bar=True)
        #self.log('train_auroc', self.auroc, on_step=False,on_epoch=True, prog_bar=True)
        
        #self.file_manager.logger.info(f'[Train] loss: {loss.item()}, accuracy: {self.accuracy.compute()}, f1: {self.f1_score.compute()}, auroc: {self.auroc.compute()}')
        #self.file_manager.logger.info(f'[Train] loss: {loss.item()}, accuracy: {self.accuracy.compute()}, f1: {self.f1_score.compute()}')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.accuracy(preds, labels)
        self.f1_score(preds, labels)
        #self.file_manager.logger.info(f"Shapes - probs: {probs.shape}, labels: {labels.shape}")
        #self.auroc(probs, labels)

        self.log('val_accuracy', self.accuracy, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1_score, on_epoch=True, prog_bar=True)
        #self.log('val_auroc', self.auroc, on_epoch=True, prog_bar=True)

        #self.file_manager.logger.info(f'[Val] loss: {loss.item()}, accuracy: {self.accuracy.compute()}, f1: {self.f1_score.compute()}, auroc: {self.auroc.compute()}')
        #self.file_manager.logger.info(f'[Val] loss: {loss.item()}, accuracy: {self.accuracy.compute()}, f1: {self.f1_score.compute()}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            'monitor': 'val_loss',  # Specifies the metric to monitor for scheduling decisions
        }
        
        return [optimizer], [scheduler]