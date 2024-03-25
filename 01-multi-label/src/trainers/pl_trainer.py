from pytorch_lightning import LightningModule
from torchmetrics.classification import MultilabelF1Score
from torchmetrics import ConfusionMatrix
import torch
import torch.nn.functional as F
from utils import FileManager, ModelConfig

torch.backends.cudnn.benchmark = True


class MultiLabelModelTrainer(LightningModule):
    def __init__(self, file_manager: FileManager, model_config: ModelConfig, model, num_labels, labels, criterion, learning_rate=2e-5):
        super().__init__()
        self.file_manager = file_manager
        self.model = model
        self.model_config = model_config
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.log_step_interval = 10

        self.conf_matrix = ConfusionMatrix(num_labels=self.num_labels, task='multilabel')
        self.f1_score = MultilabelF1Score(
            num_labels=num_labels, threshold=0.5, average='macro')

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def step(self, batch):
        if self.model_config.model == 'densenet':
            pixel_values = batch['img']
            labels = batch['lab']
        else:
            pixel_values = batch['pixel_values']
            labels = batch['labels']
        logits = self(pixel_values)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        # Update metrics
        f1 = self.f1_score(logits, labels)

        # Log metrics
        if batch_idx % self.log_step_interval == 0:
            self.log('train_f1', self.f1_score.compute(), on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'f1': f1}

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        # Update metrics
        f1 = self.f1_score(logits, labels)

        # Log metrics
        if batch_idx % self.log_step_interval == 0:
            self.log('val_f1', self.f1_score.compute(), on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_f1': f1}

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        f1 = self.f1(logits, labels)
        self.log('test_loss', loss)
        self.log('test_f1', f1)
        preds = torch.argmax(logits, dim=1)
        
        # Update confusion matrix
        self.conf_matrix.update(preds, labels.argmax(dim=1))
        
        return {'test_loss': loss, 'test_f1': f1}
    
    def test_epoch_end(self, outputs):
        # Compute the final confusion matrix for the test set
        final_conf_matrix = self.conf_matrix.compute()
        self.logger.experiment.add_image("Confusion Matrix", final_conf_matrix, self.current_epoch)
        
        # Make sure to reset the metric for future test runs
        self.conf_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer), 'monitor': 'val_loss'}

        return [optimizer], [scheduler]

    def save_model(self):
        script = self.to_torchscript()
        torch.jit.save(script, "model.pt")

    def validation_epoch_end(self, outputs):
        # Aggregate validation loss and F1 score
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()

        # Log the average validation loss and F1 score
        self.log('avg_val_loss', avg_val_loss,
                on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_val_f1', avg_val_f1, on_epoch=True,
                prog_bar=True, logger=True)

        self.file_manager.logger.info(
            f'Validation loss: {avg_val_loss}, Validation F1: {avg_val_f1}'
        )
        
        return {'avg_val_loss': avg_val_loss, 'avg_val_f1': avg_val_f1}
