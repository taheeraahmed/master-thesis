from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy, MultilabelF1Score, AUROC
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
        self.f1_score(logits, labels)

        # Log metrics
        if batch_idx % self.log_step_interval == 0:
            self.log('train_f1', self.f1_score.compute(), on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

        # Update metrics
        self.f1_score(logits, labels)

        # Log metrics
        if batch_idx % self.log_step_interval == 0:
            self.log('val_f1', self.f1_score.compute(), on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer), 'monitor': 'val_loss'}

        return [optimizer], [scheduler]

    def save_model(self):
        script = self.to_torchscript()
        torch.jit.save(script, "model.pt")
