import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
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
        print(class_weights.size())
        print(len(self.classnames))
        assert class_weights.ndim == 1, "class_weights must be a 1D tensor"
        assert len(class_weights) == len(self.classnames), "The length of class_weights must match the number of classes"
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
        return nn.CrossEntropyLoss()

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
            logits = logits.to(self.device)
            targets = labels.to(self.device)
            # compute loss
            loss = self.criterion(logits, targets)
            # backward pass and optimization
            loss.backward()
            self.optimizer.step()
            # accumulate training loss
            train_loss += loss.item()
            _, predicted_classes = torch.max(logits, 1)
            predicted_classes = predicted_classes.cpu().numpy()
            targets = targets.cpu().numpy()
            train_outputs.append(predicted_classes)
            train_targets.append(targets)
            targets_indices = torch.argmax(torch.from_numpy(targets), dim=1)
            # calculate and accumulate accuracy
            train_correct_predictions += np.sum(predicted_classes == targets_indices.numpy())
            train_total_predictions += len(targets)

            if i % 2 == 0:
                img_grid = torchvision.utils.make_grid(inputs)
                self.writer.add_image(
                    f'Epoch {epoch}/four_xray_images', img_grid)

        # Concatenate all outputs and targets for overall metrics calculation
        train_outputs = np.concatenate(train_outputs)
        train_targets = np.concatenate(train_targets)

        # Calculate overall metrics for training
        avg_train_loss = train_loss / len(train_dataloader)
        # If train_targets is a tensor and one-hot encoded
        if train_targets.ndim > 1:
            train_targets = torch.from_numpy(train_targets)
            train_targets_indices = torch.argmax(train_targets, dim=1)
            train_targets = train_targets_indices.cpu().numpy()  # Convert to NumPy array if not already

        train_accuracy = np.mean(train_outputs == train_targets_indices)
        train_f1 = f1_score(train_targets, train_outputs, average='macro')

        # Log training metrics
        self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        self.writer.add_scalar('F1/Train', train_f1, epoch)
        self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        # Optional: Log class-wise F1 scores for detailed analysis
        for cls_idx, cls_name in enumerate(self.classnames):
            cls_f1 = f1_score(train_targets, train_outputs, labels=[cls_idx], average='macro')
            self.writer.add_scalar(f'F1_classes/Train/{cls_name}', cls_f1, epoch)

        # Get the current learning rate from the scheduler
        current_lr = self.scheduler.get_last_lr()[0]
        # Log the learning rate
        self.writer.add_scalar('Learning rate', current_lr, epoch)
        # Step the scheduler
        self.scheduler.step()

        self.logger.info(
            f'[Train] Epoch {epoch+1} - Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}')

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
                targets_indices = torch.argmax(
                    torch.from_numpy(targets), dim=1)
                val_correct_predictions += np.sum(
                    predicted_classes == targets_indices.numpy())
                val_total_predictions += targets.size
                # For metrics calculation (e.g., F1 score, precision, recall), append predictions and targets
                val_outputs.append(predicted_classes)
                val_targets.append(targets)

        # Assuming you want to compute metrics outside the loop
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)

        # calculate average metrics for validation
        avg_val_loss = val_loss / len(validation_dataloader)

        # If train_targets is a tensor and one-hot encoded
        if val_targets.ndim > 1:
            val_targets = torch.from_numpy(val_targets)
            val_targets_indices = torch.argmax(val_targets, dim=1)
            # Convert to NumPy array if not already
            val_targets = val_targets_indices.cpu().numpy()

        val_accuracy = np.mean(val_outputs == val_targets_indices)
        val_f1 = f1_score(val_targets, val_outputs, average='macro')

        # Log training metrics
        self.writer.add_scalar('Loss/Train', avg_val_loss, epoch)
        self.writer.add_scalar('F1/Train', val_f1, epoch)
        self.writer.add_scalar('Accuracy/Train', val_accuracy, epoch)

        for cls_idx, cls_name in enumerate(self.classnames):
            cls_f1 = f1_score(val_targets, val_outputs,
                              labels=[cls_idx], average='macro')
            self.writer.add_scalar(
                f'F1_classes/Train/{cls_name}', cls_f1, epoch)

        # Get the current learning rate from the scheduler
        current_lr = self.scheduler.get_last_lr()[0]
        # Log the learning rate
        self.writer.add_scalar('Learning rate', current_lr, epoch)
        # Step the scheduler
        self.scheduler.step()

        self.logger.info(
            f'[Validation] Epoch {epoch+1} - Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}')
