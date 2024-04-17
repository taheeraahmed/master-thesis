from pytorch_lightning import LightningModule
from torchmetrics.classification import MultilabelF1Score
from torchmetrics import AUROC, ConfusionMatrix
import torch
from utils import FileManager
import onnx
from models import ModelConfig, set_optimizer

torch.backends.cudnn.benchmark = True


class MultiLabelLightningModule(LightningModule):
    def __init__(self, model_config: ModelConfig, file_manager: FileManager):
        super().__init__()
        self.model_config = model_config
        self.model = model_config.model
        self.model_arg = model_config.model_arg
        self.file_manager = file_manager
        self.criterion = model_config.criterion
        self.learning_rate = model_config.learning_rate
        self.num_labels = model_config.num_labels
        self.log_step_interval = 100

        self.f1_score = MultilabelF1Score(
            num_labels=self.num_labels, threshold=0.5, average='macro')

        self.f1_score_micro = MultilabelF1Score(
            num_labels=self.num_labels, threshold=0.5, average='micro')
        
        self.auroc = AUROC(
            task="multilabel",
            num_labels=self.num_labels,
            average="macro",
        )

    def forward(self, pixel_values):
        return self.model(pixel_values)

    def step(self, batch):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        if self.model_arg == 'swin':
            logits = self.forward(pixel_values.logit)
        else:
            logits = self.forward(pixel_values)

        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        # Update metrics
        f1 = self.f1_with_sigmoid(logits, labels)
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)

        # Log metrics
        if batch_idx % self.log_step_interval == 0:
            self.log('train_f1', f1, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('train_f1_micro', f1_micro, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('train_auroc', auroc, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss', loss, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        f1 = self.f1_with_sigmoid(logits, labels)
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)

        if batch_idx % self.log_step_interval == 0:
            self.log('val_f1', f1, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1_micro', f1_micro, on_step=True,
                        on_epoch=True, prog_bar=True, logger=True)
            self.log('val_auroc', auroc, on_step=True,
                        on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss', loss, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        f1 = self.f1_with_sigmoid(logits, labels)
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)

        self.log('test_loss', loss)
        self.log('test_f1', f1)
        self.log('test_f1_micro', f1_micro)
        self.log('test_auroc', auroc)

        return {'test_loss': loss, 'test_f1': f1, 'test_f1_micro': f1_micro}
    
    def on_test_end(self):
        self.save_model()

    def configure_optimizers(self):
        optimizer = set_optimizer(self.model_config)
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10), 'monitor': 'val_loss'}
        #scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 'monitor': 'val_loss'}
        return [optimizer], [scheduler]

    def save_model(self):
        img_size = self.model_config.img_size
        self.to_onnx(f"{self.file_manager.model_ckpts_folder}/test-model.onnx", input_sample=torch.randn(1, 3, img_size, img_size))

    def f1_with_sigmoid(self, logits, labels):
        preds = torch.sigmoid(logits)
        return self.f1_score(preds, labels)

    def f1_micro_with_sigmoid(self, logits, labels):
        preds = torch.sigmoid(logits)
        return self.f1_score_micro(preds, labels)
    
    def auroc_with_sigmoid(self, logits, labels):
        preds = torch.sigmoid(logits)
        return self.auroc(preds, labels.type(torch.int32))