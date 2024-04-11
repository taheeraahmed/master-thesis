from pytorch_lightning import LightningModule
from torchmetrics.classification import MultilabelF1Score
from torchmetrics import AUROC
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import io
import torchvision
from utils import FileManager
from models import ModelConfig

torch.backends.cudnn.benchmark = True


class MultiLabelLightningModule(LightningModule):
    def __init__(self, model_config: ModelConfig, file_manager: FileManager):
        super().__init__()
        self.model = model_config.model
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

        preds = torch.sigmoid(logits) > 0.5
        if batch_idx == 0:  # For example, just visualize for the first batch
            self.visualize_predictions(batch, preds, batch_idx)

        self.log('test_loss', loss)
        self.log('test_f1', f1)
        self.log('test_f1_micro', f1_micro)
        self.log('test_auroc', auroc)

        self.file_manager.logger.info(f"Test loss: {loss}")
        self.file_manager.logger.info(f"Test f1_macro: {f1}")
        self.file_manager.logger.info(f"Test f1_micro: {f1_micro}")
        self.file_manager.logger.info(f"Test auroc: {auroc}")
        
        return {'test_loss': loss, 'test_f1': f1, 'test_f1_micro': f1_micro}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer), 'monitor': 'val_loss'}
        return [optimizer], [scheduler]

    def save_model(self):
        torch.save(self.model, f"{self.file_mangager.model}model.pt")
        #script = self.to_torchscript()
        #torch.jit.save(script, "model.pt")

    def f1_with_sigmoid(self, logits, labels):
        preds = torch.sigmoid(logits)
        return self.f1_score(preds, labels)

    def f1_micro_with_sigmoid(self, logits, labels):
        preds = torch.sigmoid(logits)
        return self.f1_score_micro(preds, labels)
    
    def auroc_with_sigmoid(self, logits, labels):
        preds = torch.sigmoid(logits)
        return self.auroc(preds, labels.type(torch.int32))
    
    def visualize_predictions(self, batch, preds, batch_idx):
        # Assuming 'pixel_values' are your images in the batch
        images, true_labels = batch['pixel_values'], batch['labels']
        fig, axs = plt.subplots(nrows=1, ncols=len(images), figsize=(15, 5))
        
        for i, (img, pred, label) in enumerate(zip(images, preds, true_labels)):
            img = img.cpu().numpy().transpose(1, 2, 0)  # Assuming images are in CxHxW format
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
            axs[i].imshow(img)
            axs[i].set_title(f"Pred: {pred}, True: {label}")
            axs[i].axis('off')
        
        plt.savefig(f"{self.file_manager.image_folder}/sample_preds_{batch_idx}.png")
        plt.close()
