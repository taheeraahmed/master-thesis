from pytorch_lightning import LightningModule
from torchmetrics.classification import MultilabelF1Score
from torchmetrics import AUROC
import torch
from utils import FileManager
import onnx
import csv
import os
from models import ModelConfig, set_optimizer, set_scheduler

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
        self.labels  = model_config.labels
        self.log_step_interval = 100

        self.test_results=[]

        self.f1_score = MultilabelF1Score(
            num_labels=self.num_labels, threshold=0.5, average='macro')

        self.f1_score_micro = MultilabelF1Score(
            num_labels=self.num_labels, threshold=0.5, average='micro')
        
        self.auroc = AUROC(
            task="multilabel",
            num_labels=self.num_labels,
            average="macro",
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
        if mode == 'test' and pixel_values.dim() == 5:  # Assuming shape [batch, crops, channels, height, width]
            bs, n_crops, c, h, w = pixel_values.size()
            pixel_values = pixel_values.view(-1, c, h, w)  # Flatten the crops into the batch dimension
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
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)
        auroc_classwise = self.auroc_classwise_with_sigmoid(logits, labels.type(torch.int32))

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
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        mode='val'

        loss, logits, labels = self.step(batch, mode=mode)
        f1 = self.f1_with_sigmoid(logits, labels)
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)
        auroc_classwise = self.auroc_classwise_with_sigmoid(logits, labels.type(torch.int32))

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
        
    def test_step(self, batch, batch_idx):
        mode = 'test'
        loss, logits, labels = self.step(batch, mode=mode)
        f1 = self.f1_with_sigmoid(logits, labels)
        f1_micro = self.f1_micro_with_sigmoid(logits, labels)
        auroc = self.auroc_with_sigmoid(logits, labels)
        auroc_classwise = self.auroc_classwise_with_sigmoid(logits, labels.type(torch.int32))
        
        self.calc_classwise_auroc(auroc_classwise, mode=mode)
        self.log('test_loss', loss)
        self.log('test_f1', f1)
        self.log('test_f1_micro', f1_micro)
        self.log('test_auroc', auroc)

        self.test_results.append({
            'loss': loss.item(),
            'f1': f1.item(),
            'f1_micro': f1_micro.item(),
            'auroc': auroc.item()
        })

        return {'test_loss': loss, 'test_f1': f1, 'test_f1_micro': f1_micro}
    
    def on_test_end(self):
        if not self.model_config.fast_dev_run:
            self.save_model()
            self.save_metrics_to_csv()

    def configure_optimizers(self):
        optimizer = set_optimizer(self.model_config)
        scheduler = set_scheduler(self.model_config, optimizer)
        return [optimizer], [scheduler]
    
    def save_metrics_to_csv(self):
        csv_file_path = os.path.join(self.file_manager.root, 'test_metrics.csv')

        model_info = {
            'experiment_name': self.model_config.experiment_name,
            'model_name': self.model_arg,
            'loss_arg': self.model_config.loss_arg,
            'batch_size': self.model_config.batch_size,
            'scheduler_arg': self.model_config.scheduler_arg,
            'optimizer_arg': self.model_config.optimizer_arg,
            'learning_rate': self.model_config.learning_rate
        }

        if self.test_results:
            aggregated_results = {key: sum([batch[key] for batch in self.test_results]) / len(self.test_results) 
                                for key in self.test_results[0].keys()}
        else:
            aggregated_results = {}

        final_results = {**model_info, **aggregated_results}
        file_exists_and_not_empty = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0
        
        try:
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=final_results.keys())
                if not file_exists_and_not_empty:
                    writer.writeheader()
                writer.writerow(final_results)
            self.file_manager.logger.info(f"Test metrics saved to {csv_file_path}")
        except Exception as e:
            self.file_manager.logger.error(f"Error saving test metrics to {csv_file_path}: {e}")

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
    
    def auroc_classwise_with_sigmoid(self, logits, labels):
        preds = torch.sigmoid(logits)
        return self.auroc_classwise(preds, labels.type(torch.int32))
    
    def calc_classwise_auroc(self, auroc_classwise, mode):
        for label_name, score in zip(self.labels, auroc_classwise):
            label_name = label_name.lower()
            self.log(f'{mode}_auroc_{label_name}', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)

# class MultiLabelInferenceModel(LightningModule):
#     def __init__(self, model, criterion, learning_rate, num_labels, img_size, labels):
#         super().__init__()
#         self.model = model
#         self.criterion = criterion
#         self.learning_rate = learning_rate
#         self.num_labels = num_labels
#         self.log_step_interval = 100
#         self.img_size = img_size
#         self.labels = labels

#         self.f1_score = MultilabelF1Score(num_labels=self.num_labels, threshold=0.5, average='macro')
#         self.f1_score_micro = MultilabelF1Score(num_labels=self.num_labels, threshold=0.5, average='micro')
#         self.auroc = AUROC(task="multilabel", num_labels=self.num_labels, average="macro")
#         self.auroc_classwise = AUROC(
#             task="multilabel",
#             num_labels=self.num_labels,
#             average=None,
#         )


#     def forward(self, pixel_values):
#         return self.model(pixel_values)

#     def step(self, batch):
#         pixel_values = batch['pixel_values']
#         labels = batch['labels']
        
#         logits = self.forward(pixel_values)

#         loss = self.criterion(logits, labels)
#         return loss, logits, labels

#     def training_step(self, batch, batch_idx):
#         loss, logits, labels = self.step(batch)
#         self.log('train_loss', loss, on_step=True,
#                  on_epoch=True, prog_bar=True, logger=True)
#         # Update metrics
#         f1 = self.f1_with_sigmoid(logits, labels)
#         f1_micro = self.f1_micro_with_sigmoid(logits, labels)
#         auroc = self.auroc_with_sigmoid(logits, labels)
#         auroc_classwise = self.auroc_classwise_with_sigmoid(logits, labels.type(torch.int32))

#         if batch_idx % self.log_step_interval == 0:
#             self.calc_classwise_auroc(auroc_classwise, 'train')
#             self.log('train_f1', f1, on_step=True,
#                      on_epoch=True, prog_bar=True, logger=True)
#             self.log('train_f1_micro', f1_micro, on_step=True,
#                      on_epoch=True, prog_bar=True, logger=True)
#             self.log('train_auroc', auroc, on_step=True,
#                      on_epoch=True, prog_bar=True, logger=True)
#             self.log('train_loss', loss, on_step=True,
#                      on_epoch=True, prog_bar=True, logger=True)
        
#         return loss
        
#     def validation_step(self, batch, batch_idx):
        
#         loss, logits, labels = self.step(batch)
#         f1 = self.f1_with_sigmoid(logits, labels)
#         f1_micro = self.f1_micro_with_sigmoid(logits, labels)
#         auroc = self.auroc_with_sigmoid(logits, labels)
#         auroc_classwise = self.auroc_classwise_with_sigmoid(logits, labels.type(torch.int32))

#         if batch_idx % self.log_step_interval == 0:
#             self.calc_classwise_auroc(auroc_classwise, 'val')
#             self.log('val_f1', f1, on_step=True,
#                      on_epoch=True, prog_bar=True, logger=True)
#             self.log('val_f1_micro', f1_micro, on_step=True,
#                         on_epoch=True, prog_bar=True, logger=True)
#             self.log('val_auroc', auroc, on_step=True,
#                         on_epoch=True, prog_bar=True, logger=True)
#             self.log('val_loss', loss, on_step=True,
#                      on_epoch=True, prog_bar=True, logger=True)
            
#     def calc_classwise_auroc(self, auroc_classwise):
#         for label_name, score in zip(self.label_names, auroc_classwise):
#             self.log(f'train_auroc_{label_name}', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)

#     def test_step(self, batch, batch_idx):
#         loss, logits, labels = self.step(batch)
#         f1 = self.f1_with_sigmoid(logits, labels)
#         f1_micro = self.f1_micro_with_sigmoid(logits, labels)
#         auroc = self.auroc_with_sigmoid(logits, labels)
#         auroc_classwise = self.auroc_classwise_with_sigmoid(logits, labels.type(torch.int32))

#         self.calc_classwise_auroc(auroc_classwise, 'test')        
#         self.log('test_loss', loss)
#         self.log('test_f1', f1)
#         self.log('test_f1_micro', f1_micro)
#         self.log('test_auroc', auroc)

#         return {'test_loss': loss, 'test_f1': f1, 'test_f1_micro': f1_micro}
    
#     def on_test_end(self):
#         self.save_model()

#     def configure_optimizers(self):
#         optimizer = self.optimizer
#         scheduler = self.scheduler
#         return [optimizer], [scheduler]

#     def save_model(self):
#         self.file_manager.logger.info(onnx.__version__)  # This will print the version of ONNX installe

#         self.to_onnx(f"./test-model.onnx", input_sample=torch.randn(1, 3, self.img_size, self.img_size))

#     def f1_with_sigmoid(self, logits, labels):
#         preds = torch.sigmoid(logits)
#         return self.f1_score(preds, labels)

#     def f1_micro_with_sigmoid(self, logits, labels):
#         preds = torch.sigmoid(logits)
#         return self.f1_score_micro(preds, labels)
    
#     def auroc_with_sigmoid(self, logits, labels):
#         preds = torch.sigmoid(logits)
#         return self.auroc(preds, labels.type(torch.int32))

#     def auroc_classwise_with_sigmoid(self, logits, labels):
#         preds = torch.sigmoid(logits)
#         return self.auroc_classwise(preds, labels.type(torch.int32))