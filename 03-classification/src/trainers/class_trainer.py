import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import time
from utils import calculate_idun_time_left, FocalLoss
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

    def __init__(self, model, model_name, model_output_folder, logger, optimizer, classnames, log_dir='runs', loss_arg='wce'):
        self.model = model
        self.model_name = model_name
        self.model_output_folder = model_output_folder
        self.logger = logger
        self.optimizer = optimizer
        self.classnames = classnames
        self.loss_arg = loss_arg
        self.device = None

        self.device = self._get_device()                    # device
        self.scheduler = torch.optim.lr_scheduler.StepLR(   # scheduler
            self.optimizer, step_size=5, gamma=0.1)
        self.class_weights = self._get_class_weights()      # class weights
        self.criterion = self._get_loss()                   # loss function

        # best validation F1 score, for checkpointing
        self.best_val_f1 = 0.0
        # tensorboard writer
        self.writer = SummaryWriter(log_dir)

    def _get_class_weights(self):
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(self.classnames), y=self.classnames)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        assert class_weights.ndim == 1, "class_weights must be a 1D tensor"
        assert len(class_weights) == len(
            self.classnames), "The length of class_weights must match the number of classes"
        return class_weights.to(self.device)

    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            self.logger.warning('GPU unavailable, using CPU instead.')
        self.device = device
        self.model.to(self.device)
        return device

    def _get_loss(self):
        if self.loss_arg == 'ce':
            return nn.CrossEntropyLoss()
        elif self.loss_arg == 'wfl':
            return FocalLoss(alpha=self.class_weights)
        else:
            raise ValueError('Invalid loss function')

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

    def _calc_auc(self, targets, target_indices, outputs, epoch, mode):
        # convert and reshape if necessary
        outputs_tensor = torch.tensor(outputs, dtype=torch.float).view(-1, 1)
        outputs_probs = torch.nn.functional.softmax(
            outputs_tensor, dim=1).numpy()

        if targets.ndim == 1:
            # binarize labels in a one-vs-all fashion
            n_classes = np.unique(target_indices).size
            targets_binarized = label_binarize(
                target_indices, classes=range(n_classes))
        else:
            targets_binarized = targets  # assuming train_targets is already one-hot encoded

        # calculate mean AUC
        mean_auc = 0
        try:
            mean_auc = roc_auc_score(
                targets_binarized, outputs_probs, average='macro', multi_class='ovr')
        except ValueError as e:
            self.logger.error(f"Error calculating mean AUC: {e}")

        # calculate class-wise AUC
        for cls_idx, cls_name in enumerate(self.classnames):
            try:
                # adjusted to use outputs_probs
                class_auc = roc_auc_score(
                    targets_binarized[:, cls_idx], outputs_probs[:, 0], multi_class='ovr')
                self.writer.add_scalar(
                    f'AUC/{mode}/{cls_name}', class_auc, epoch)
            except ValueError as e:
                self.logger.error(f"Error calculating AUC for {cls_name}: {e}")

        return mean_auc

    def _train_epoch(self, train_dataloader, epoch, model_arg):
        self.model.train()

        # vars to store metrics for training
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0
        train_outputs = []
        train_targets = []

        train_loop = tqdm(train_dataloader, leave=True)

        for i, batch in enumerate(train_loop):
            inputs, targets = batch["img"].to(
                self.device), batch["lab"].to(self.device)
    
            self.optimizer.zero_grad()
            # forward pass
            outputs = self.model(inputs)
            logits = outputs

            # calculate loss
            probabilities = F.softmax(logits, dim=1)  # Apply softmax to logits
            # Now pass probabilities to your loss function
            loss = self.criterion(probabilities, targets)
            # backward pass and optimization
            loss.backward()
            self.optimizer.step()
            # accumulate training loss
            train_loss += loss.item()

            # get the predicted class indices with the highest probability
            _, predicted_classes = torch.max(logits, 1)
            predicted_classes = predicted_classes.cpu().numpy()
            targets = targets.cpu().numpy()
            train_outputs.append(predicted_classes)
            train_targets.append(targets)
            targets_indices = torch.argmax(torch.from_numpy(targets), dim=1)
            # calculate and accumulate accuracy
            train_correct_predictions += np.sum(
                predicted_classes == targets_indices.numpy())
            train_total_predictions += len(targets)

            if i % 2 == 0:
                img_grid = torchvision.utils.make_grid(inputs)
                self.writer.add_image(
                    f'Epoch {epoch}/four_xray_images', img_grid)

        # concatenate all outputs and targets for overall metrics calculation
        train_outputs = np.concatenate(train_outputs)
        train_targets = np.concatenate(train_targets)

        # calculate overall metrics for training
        avg_train_loss = train_loss / len(train_dataloader)

        # convert train_targets to a PyTorch tensor before using torch.argmax()
        if train_targets.ndim > 1:
            # convert numpy.ndarray to PyTorch tensor
            train_targets_tensor = torch.tensor(train_targets)
            train_targets_indices = torch.argmax(train_targets_tensor, dim=1)
        else:
            # train_targets already a tensor of class indices, ensure it's in the correct format
            train_targets_indices = torch.tensor(train_targets).long()

        # calculate AUC
        train_mean_auc = self._calc_auc(train_targets, train_targets_indices,
                                        train_outputs, epoch, mode='Train')

        if not isinstance(train_targets_indices, np.ndarray):
            train_targets_indices = train_targets_indices.cpu().numpy()

        train_accuracy = np.mean(train_outputs == train_targets_indices)

        train_f1 = f1_score(train_targets_indices,
                            train_outputs, average='macro')

        # log training metrics
        self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        self.writer.add_scalar('F1/Train', train_f1, epoch)
        self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        # class-wise F1 scores for detailed analysis
        for cls_idx, cls_name in enumerate(self.classnames):
            cls_f1 = f1_score(train_targets_indices, train_outputs,
                              labels=[cls_idx], average='macro')

            self.writer.add_scalar(
                f'F1_classes/Train/{cls_name}', cls_f1, epoch)

        # current learning rate from the scheduler
        current_lr = self.scheduler.get_last_lr()[0]
        # log the learning rate
        self.writer.add_scalar('Learning rate', current_lr, epoch)
        # step the scheduler
        self.scheduler.step()

        self.logger.info(
            f'[Train] Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, mAUC: {train_mean_auc:.4f}')

    def _validate_epoch(self, validation_dataloader, epoch, model_arg):
        self.model.eval()

        # vars to store metrics
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        val_outputs = []
        val_targets = []

        # storage for class-wise metrics
        val_class_auc = {cls_name: [] for cls_name in self.classnames}

        with torch.no_grad():
            val_loop = tqdm(validation_dataloader, leave=True)
            for i, batch in enumerate(val_loop):
                inputs, targets = batch["img"].to(
                    self.device), batch["lab"].to(self.device)

                # forward pass
                outputs = self.model(inputs)
                logits = outputs if model_arg == 'densenet' else outputs.logits

                # calculate loss
                probabilities = F.softmax(logits, dim=1)  
                loss = self.criterion(probabilities, targets)

                # accumulate validation loss
                val_loss += loss.item()

                # get the predicted class indices with the highest probability
                _, predicted_classes = torch.max(logits, 1)
                # convert to numpy arrays for comparison
                predicted_classes = predicted_classes.cpu().numpy()
                targets = targets.cpu().numpy()

                # calculate and accumulate accuracy
                targets_indices = torch.argmax(
                    torch.from_numpy(targets), dim=1)
                val_correct_predictions += np.sum(
                    predicted_classes == targets_indices.numpy())
                val_total_predictions += targets.size
                val_outputs.append(predicted_classes)
                val_targets.append(targets)

        # assuming you want to compute metrics outside the loop
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)

        # calculate average metrics for validation
        avg_val_loss = val_loss / len(validation_dataloader)

        # one-hot encoding of labels
        if val_targets.ndim > 1:
            # convert numpy.ndarray to PyTorch tensor
            val_targets_tensor = torch.tensor(val_targets)
            val_targets_indices = torch.argmax(val_targets_tensor, dim=1)
        else:
            val_targets_indices = torch.tensor(val_targets).long()

        if not isinstance(val_targets_indices, np.ndarray):
            val_targets_indices = val_targets_indices.cpu().numpy()

        val_mean_auc = self._calc_auc(
            val_targets, val_targets_indices, val_outputs, epoch, mode='Val')

        val_accuracy = np.mean(val_outputs == val_targets_indices)
        val_f1 = f1_score(val_targets_indices, val_outputs, average='macro')

        self.writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        self.writer.add_scalar('F1/Val', val_f1, epoch)
        self.writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

        # class-wise F1 score for validation
        for cls_idx, cls_name in enumerate(self.classnames):
            cls_f1 = f1_score(val_targets_indices, val_outputs,
                              labels=[cls_idx], average='macro')

            self.writer.add_scalar(
                f'F1_classes/Val/{cls_name}', cls_f1, epoch)

        # Get the current learning rate from the scheduler
        current_lr = self.scheduler.get_last_lr()[0]
        # Log the learning rate
        self.writer.add_scalar('Learning rate', current_lr, epoch)
        # Step the scheduler
        self.scheduler.step()

        self.logger.info(
            f'[Validation] Epoch {epoch+1} - Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, mAUC: {val_mean_auc:.4f}')
