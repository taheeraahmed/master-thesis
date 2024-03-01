import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initializes the Focal Loss.

        Args:
            alpha (float, optional): Weighting factor for the class imbalance. Defaults to 1.0.
            gamma (float, optional): Focusing parameter to adjust the rate at which easy examples are down-weighted.
                                      Defaults to 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                       'none': no reduction will be applied,
                                       'mean': the sum of the output will be divided by the number of elements in the output,
                                       'sum': the output will be summed. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of the focal loss.

        Args:
            inputs (torch.Tensor): Probabilities for each class, obtained after applying softmax or sigmoid to the logits.
                                   Shape [batch_size, num_classes] for multi-class classification or [batch_size, 1] for binary.
            targets (torch.Tensor): Ground truth labels, one-hot encoded. Shape [batch_size, num_classes] for multi-class.

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Ensure inputs are probabilities and targets are one-hot encoded
        if not (inputs.size() == targets.size()):
            raise ValueError("Size mismatch between inputs and targets")

        # Calculate the cross entropy loss for each class
        # Adding epsilon to avoid log(0)
        ce_loss = -targets * torch.log(inputs + 1e-8)

        # Calculate the focal loss adjustment factor
        focal_loss_adjustment = self.alpha * (1 - inputs) ** self.gamma

        # Apply adjustment factor to the cross entropy loss
        focal_loss = focal_loss_adjustment * ce_loss

        # Reduce the loss based on the reduction parameter
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

# Example usage
# Assuming `logits` is your model's output and `labels` is your ground truth labels, one-hot encoded
# model_output = model(inputs)  # shape [batch_size, num_classes]
# probabilities = F.softmax(model_output, dim=1)  # Convert logits to probabilities
# focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
# loss = focal_loss_fn(probabilities, labels)
