from pytorch_lightning import LightningModule
from torchmetrics.classification import MultilabelF1Score
from torchmetrics import AUROC
import torch
import onnx
import csv
import os


torch.backends.cudnn.benchmark = True


class MultiLabelLightningModule(LightningModule):
    def __init__(self, model, criterion, learning_rate, num_labels, labels, optimizer_func, scheduler_func, model_ckpts_folder, file_logger=None, root_path="./", model_name="model", experiment_name="experiment", img_size=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.labels = labels
        self.model_ckpts_folder = model_ckpts_folder
        self.root_path = root_path
        self.optimizer_func = optimizer_func
        self.scheduler_func = scheduler_func
        self.log_step_interval = 100
        self.file_logger = file_logger
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.img_size = img_size

        self.test_results = []

        self.f1_score = MultilabelF1Score(
            num_labels=self.num_labels, threshold=0.5, average='macro')

        self.f1_score_micro = MultilabelF1Score(
            num_labels=self.num_labels, threshold=0.5, average='micro')

        self.auroc = AUROC(
            task="multilabel",
            num_labels=self.num_labels,
            average="macro",
        )

        self.auroc_micro = AUROC(
            task="multilabel",
            num_labels=self.num_labels,
            average="micro",
        )

        self.auroc_classwise = AUROC(
            task="multilabel",
            num_labels=self.num_labels,
            average=None,
        )

    def forward(self, pixel_values):
        return self.model(pixel_values)

    def step(self, batch, mode='train_val'):
        pixel_values = batch['pixel_values']
        labels = batch['labels']

        # Check if we are in test mode and if the input is for TTA
        # Assuming shape [batch, crops, channels, height, width]
        if mode == 'test' and pixel_values.dim() == 5:
            bs, n_crops, c, h, w = pixel_values.size()
            # Flatten the crops into the batch dimension
            pixel_values = pixel_values.view(-1, c, h, w)
        else:
            n_crops = 1

        logits = self.forward(pixel_values)

        # Average predictions across crops only if TTA is applied
        if mode == 'test' and n_crops > 1:
            logits = logits.view(bs, n_crops, -1).mean(1)

        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        mode = 'train'
        loss, logits, labels = self.step(batch, mode=mode)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        # Update metrics
        f1 = self.f1_with_sigmoid(logits, labels)
        auroc_micro = self.auroc_micro_with_sigmoid(logits, labels)
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)
        auroc_classwise = self.auroc_classwise_with_sigmoid(
            logits, labels.type(torch.int32))

        # Log metrics
        if batch_idx % self.log_step_interval == 0:
            self.calc_classwise_auroc(auroc_classwise, mode=mode)
            self.log('train_f1', f1, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('train_f1_micro', f1_micro, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('train_auroc', auroc, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss', loss, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('train_auroc_micro', auroc_micro, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True,
                on_epoch=True, prog_bar=True, logger=True)


    def validation_step(self, batch, batch_idx):
        mode = 'val'

        loss, logits, labels = self.step(batch, mode=mode)
        f1 = self.f1_with_sigmoid(logits, labels)
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc_micro = self.auroc_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)
        auroc_classwise = self.auroc_classwise_with_sigmoid(
            logits, labels.type(torch.int32))

        if batch_idx % self.log_step_interval == 0:
            self.calc_classwise_auroc(auroc_classwise, mode)
            self.log('val_f1', f1, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1_micro', f1_micro, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('val_auroc', auroc, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss', loss, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('val_auroc_micro', auroc_micro, on_step=True,
                        on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        mode = 'test'
        loss, logits, labels = self.step(batch, mode=mode)
        f1 = self.f1_with_sigmoid(logits, labels)
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc_micro = self.auroc_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)

        auroc_classwise = self.auroc_classwise_with_sigmoid(
            logits, labels.type(torch.int32))

        self.calc_classwise_auroc(auroc_classwise, mode=mode)
        self.log('test_loss', loss)
        self.log('test_f1', f1)
        self.log('test_f1_micro', f1_micro)
        self.log('test_auroc', auroc)
        self.log('test_auroc_micro', auroc_micro)

        self.test_results.append({
            'loss': loss.item(),
            'f1': f1.item(),
            'f1_micro': f1_micro.item(),
            'auroc': auroc.item()
        })

        return {'test_loss': loss, 'test_f1': f1, 'test_f1_micro': f1_micro}

    def on_test_end(self):
        self.print_metrics()

        if self.img_size is not None:
            try:
                self.save_model()
            except Exception as e:
                if self.file_logger:
                    self.file_logger.error(
                        f"Error saving model to {self.model_ckpts_folder}: {e}")
                else:
                    print(f"Error saving model to {self.model_ckpts_folder}: {e}")

    def configure_optimizers(self):
        optimizer = self.optimizer_func
        scheduler = self.scheduler_func
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_auroc",
                "frequency": 1,
                "interval": "step",
            },
        }

    def print_metrics(self):
        if self.test_results:
            final_results = {key: sum([batch[key] for batch in self.test_results]) / len(self.test_results)
                                  for key in self.test_results[0].keys()}
        else:
            final_results = {}

        if self.file_logger:
            self.file_logger.info(f"Test results: {final_results}")
        else: 
            print(final_results)

    def save_model(self):
        img_size = self.img_size
        if not (os.path.exists(f"{self.model_ckpts_folder}")):
            os.mkdir(f"{self.model_ckpts_folder}")
        
        self.to_onnx(f"{self.model_ckpts_folder}/test-model.onnx",
                        input_sample=torch.randn(1, 3, img_size, img_size))

    def f1_with_sigmoid(self, logits, labels):
        probs = torch.sigmoid(logits)
        return self.f1_score(probs, labels)

    def f1_micro_with_sigmoid(self, logits, labels):
        probs = torch.sigmoid(logits)
        return self.f1_score_micro(probs, labels)

    def auroc_with_sigmoid(self, logits, labels):
        probs = torch.sigmoid(logits)
        return self.auroc(probs, labels.type(torch.int32))
    
    def auroc_micro_with_sigmoid(self, logits, labels):
        probs = torch.sigmoid(logits)
        return self.auroc_micro(probs, labels.type(torch.int32))

    def auroc_classwise_with_sigmoid(self, logits, labels):
        probs = torch.sigmoid(logits)
        return self.auroc_classwise(probs, labels.type(torch.int32))

    def calc_classwise_auroc(self, auroc_classwise, mode):
        for label_name, score in zip(self.labels, auroc_classwise):
            label_name = label_name.lower()
            self.log(f'{mode}_auroc_{label_name}', score,
                     on_step=True, on_epoch=True, prog_bar=True, logger=True)
