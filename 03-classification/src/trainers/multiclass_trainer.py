import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score
import time
from utils import calculate_idun_time_left, WeightedFocalLoss
import torchvision
import numpy as np
from tqdm import tqdm
import os

torch.backends.cudnn.benchmark = True


class TrainerClass:
    """
    Class to handle training and validation of a multi-class classification model.
    It works with DenseNet from xrayvision library
    """

    def __init__(self, model, model_name, model_output_folder, logger, optimizer, log_dir='runs', class_weights=None, loss_arg='wce'):
        self.model = model
        self.model_name = model_name
        self.model_output_folder = model_output_folder
        self.logger = logger
        self.optimizer = optimizer
        self.classnames = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
                           'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
                           'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
        self.loss_arg = loss_arg

        self.device = self._get_device()                    # device
        self.criterion = self._get_loss()                   # loss function
        self.scheduler = torch.optim.lr_scheduler.StepLR(   # scheduler
            self.optimizer, step_size=5, gamma=0.1)
        self.class_weights = class_weights.to(self.device)  # class weights

        # best validation F1 score, for checkpointing
        self.best_val_f1 = 0.0
        # tensorboard writer
        self.writer = SummaryWriter(log_dir)

    def _get_device(self):
        if torch.cuda.is_available():
            self.model.to(self.device)
            return torch.device("cuda")
        else:
            self.logger.warning('GPU unavailable')
            return torch.device("cpu")

    def _get_loss(self):
        loss_arg = self.loss_arg

        if self.class_weights is not None:
            self.logger.info('Using weighted loss')
            assert self.class_weights.ndim == 1, "class_weights must be a 1D tensor"
            assert len(self.class_weights) == len(
                self.classnames), "The length of class_weights must match the number of classes"
            if loss_arg == 'wfl':
                return WeightedFocalLoss(alpha=self.class_weights.to(
                    self.device), gamma=2.0, reduction='mean')
            elif loss_arg == 'wce':
                return nn.CrossEntropyLoss(pos_weight=self.class_weights)
            else:
                self.logger.error("Invalid loss function")
                raise ValueError("Invalid loss function")
        else:
            self.logger.info('Using unweighted loss')
            if loss_arg == 'fl':
                return WeightedFocalLoss(
                    alpha=None, gamma=2.0, reduction='mean')
            elif loss_arg == 'bce':
                return nn.CrossEntropyLoss()
            else:
                self.logger.error("Invalid loss function")
                raise ValueError("Invalid loss function")

    def train(self, train_dataloader, validation_dataloader, num_epochs, idun_datetime_done, model_arg):
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.logger.info(f'Started epoch {epoch+1}')

            self._train_epoch(train_dataloader, epoch, model_arg)
            self._validate_epoch(validation_dataloader, epoch, model_arg)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            calculate_idun_time_left(
                epoch, num_epochs, epoch_duration, idun_datetime_done, self.logger)

        self.writer.close()

    def _save_checkpoint(self, epoch, current_val_accuracy):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1
        }

        if not os.path.exists(self.model_output_folder):
            os.makedirs(self.model_output_folder)

        checkpoint_path = os.path.join(
            self.model_output_folder, f'model_checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            f'Checkpoint saved for epoch {epoch+1} with f1 score: {current_val_accuracy}')

    def _train_epoch(self, train_dataloader, epoch, model_arg):
        self.model.train()

        # vars to store metrics for training
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0
        train_outputs = []
        train_targets = []

        # storage for class-wise metrics
        train_class_f1 = {cls_name: [] for cls_name in self.classnames}
        train_class_auc = {cls_name: [] for cls_name in self.classnames}

        train_loop = tqdm(train_dataloader, leave=True)

        for i, batch in enumerate(train_loop):
            inputs, labels = batch["img"].to(
                self.device), batch["lab"].to(self.device)
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs)

            logits = outputs if model_arg == 'densenet' else outputs.logits
            # Ensure logits are on the correct device
            logits = logits.to(self.device)
            # Ensure targets are on the correct device
            targets = labels.to(self.device)

            # compute loss
            loss = self.criterion(logits, targets)

            # backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # accumulate training loss
            train_loss += loss.item()

            # Get the predicted class indices with the highest probability
            _, predicted_classes = torch.max(logits, 1)
            # Convert to numpy arrays for comparison
            predicted_classes = predicted_classes.cpu().numpy()
            targets = targets.cpu().numpy()

            # appending for potential metrics calculation outside the loop
            train_outputs.append(predicted_classes)
            train_targets.append(targets)

            # Calculate and accumulate accuracy
            train_correct_predictions += np.sum(predicted_classes == targets)
            train_total_predictions += targets.size

            if i % 2 == 0:
                img_grid = torchvision.utils.make_grid(inputs)
                self.writer.add_image(
                    f'Epoch {epoch}/four_xray_images', img_grid)

        # concatenate all outputs and targets
        train_outputs = np.vstack(train_outputs)
        train_targets = np.vstack(train_targets)

        # calculate class-wise F1 and AUC
        for cls_idx, cls_name in enumerate(self.classnames):
            cls_f1 = f1_score(
                train_targets[:, cls_idx], train_outputs[:, cls_idx])
            train_class_f1[cls_name].append(cls_f1)

            try:
                cls_auc = roc_auc_score(
                    train_targets[:, cls_idx], train_outputs[:, cls_idx])
                train_class_auc[cls_name].append(cls_auc)
            except ValueError:
                self.logger.warning(
                    f'Error calculating AUC for class {cls_name}')

        # calculate average metrics for training
        avg_train_loss = train_loss / len(train_dataloader)
        train_f1 = f1_score(train_targets, train_outputs, average='macro')
        train_accuracy = np.mean(train_targets == train_outputs)

        # log class-wise metrics
        for cls_name in self.classnames:
            self.writer.add_scalar(
                f'F1_classes//Train/{cls_name}', np.mean(train_class_f1[cls_name]), epoch)
            if train_class_auc[cls_name]:
                self.writer.add_scalar(
                    f'AUC_classes/Train/{cls_name}', np.mean(train_class_auc[cls_name]), epoch)

        # log training metrics
        self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        self.writer.add_scalar('F1/Train', train_f1, epoch)
        self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        # calculate AUC
        try:
            train_auc = roc_auc_score(
                train_targets, train_outputs, average='macro')
            self.writer.add_scalar('AUC/Train', train_auc, epoch)
            self.logger.info(
                f'[Train] Epoch {epoch+1} - loss: {avg_train_loss}, F1: {train_f1}, auc: {train_auc}, accuracy: {train_accuracy}')
        except ValueError as e:
            self.logger.warning(
                f'Unable to calculate train AUC for epoch {epoch+1}: {e}')
            self.logger.info(
                f'[Train] Epoch {epoch+1} - loss: {avg_train_loss}, F1: {train_f1}, accuracy: {train_accuracy}')

        # Get the current learning rate from the scheduler
        # Extract the first (and likely only) element
        current_lr = self.scheduler.get_last_lr()[0]
        # Log the learning rate
        self.writer.add_scalar('Learning rate', current_lr, epoch)
        # Step the scheduler
        self.scheduler.step()

    def _validate_epoch(self, validation_dataloader, epoch, model_arg):
        self.model.eval()

        # vars to store metrics
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        val_outputs = []
        val_targets = []

        # storage for class-wise metrics
        val_class_f1 = {cls_name: [] for cls_name in self.classnames}
        val_class_auc = {cls_name: [] for cls_name in self.classnames}

        with torch.no_grad():
            val_loop = tqdm(validation_dataloader, leave=True)
            for i, batch in enumerate(val_loop):
                inputs, labels = batch["img"].to(
                    self.device), batch["lab"].to(self.device)

                # forward pass
                outputs = self.model(inputs)
                logits = outputs if model_arg == 'densenet' else outputs.logits
                # Ensure logits are on the correct device
                logits = logits.to(self.device)
                # Ensure targets are on the correct device
                targets = labels.to(self.device)

                # compute loss
                loss = self.criterion(logits, targets)

                # accumulate validation loss
                val_loss += loss.item()

                # Get the predicted class indices with the highest probability
                _, predicted_classes = torch.max(logits, 1)
                # Convert to numpy arrays for comparison
                predicted_classes = predicted_classes.cpu().numpy()
                targets = targets.cpu().numpy()

                # calculate and accumulate accuracy
                val_correct_predictions += np.sum(predicted_classes == targets)
                val_total_predictions += targets.size

                # For metrics calculation (e.g., F1 score, precision, recall), append predictions and targets
                val_outputs.append(predicted_classes)
                val_targets.append(targets)

        # Assuming you want to compute metrics outside the loop
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)

        # class-wise f1 and auc
        for cls_idx, cls_name in enumerate(self.classnames):
            cls_f1 = f1_score(val_targets[:, cls_idx], val_outputs[:, cls_idx])
            val_class_f1[cls_name].append(cls_f1)

            try:
                cls_auc = roc_auc_score(
                    val_targets[:, cls_idx], val_outputs[:, cls_idx])
                val_class_auc[cls_name].append(cls_auc)
            except ValueError:
                self.logger.warning(
                    f'Error calculating AUC for class {cls_name}')

        # calculate average metrics for validation
        avg_val_loss = val_loss / len(validation_dataloader)
        val_f1 = f1_score(targets_binary, outputs_binary, average='macro')
        val_accuracy = val_correct_predictions / val_total_predictions

        # write to tensorboard
        for cls_name in self.classnames:
            self.writer.add_scalar(
                f'F1_classes/Validation/{cls_name}', np.mean(val_class_f1[cls_name]), epoch)
            if val_class_auc[cls_name]:
                self.writer.add_scalar(
                    f'AUC_classes/Validation/{cls_name}', np.mean(val_class_auc[cls_name]), epoch)
        self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        self.writer.add_scalar('F1/Validation', val_f1, epoch)
        self.writer.add_scalar('Accuracy/Train', val_accuracy, epoch)

        # log and check if possible to calculate AUC
        try:
            val_auc = roc_auc_score(val_targets, val_outputs, average='macro')
            self.writer.add_scalar('AUC/Validation', val_auc, epoch)
            self.logger.info(
                f'[Validation] Epoch {epoch+1} - loss: {avg_val_loss}, F1: {val_f1}, auc: {val_auc}, accuracy: {val_accuracy}')
        except ValueError as e:
            self.logger.warning(
                f'Unable to calculate validation AUC for epoch {epoch+1}: {e}')
            self.logger.info(
                f'[Validation] Epoch {epoch+1} - loss: {avg_val_loss}, F1: {val_f1}, accuracy: {val_accuracy}')

        # checkpointing
        current_val_f1 = val_f1
        if current_val_f1 > self.best_val_f1:
            self.best_val_f1 = current_val_f1
            self._save_checkpoint(epoch, current_val_f1)
