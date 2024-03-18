import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Initializes the weighted multi-label focal loss.

        Parameters:
        - weight (Tensor, optional): Weights for each class.
        - gamma (float): Focusing parameter.
        - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the focal loss calculation.

        Parameters:
        - inputs (Tensor): Predictions from the model (logits before sigmoid).
        - targets (Tensor): Ground truth labels.

        Returns:
        - loss (Tensor): Calculated focal loss.
        """
        if self.weight is not None:
            self.weight = self.weight.to(inputs.device)

        # Apply sigmoid to logits
        probs = torch.sigmoid(inputs)
        # Calculate the binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate the focal loss adjustment factors
        probs_with_targets = torch.where(targets >= 0.5, probs, 1 - probs)
        focal_loss_adjustment = torch.pow((1 - probs_with_targets), self.gamma)

        # Apply weights if provided
        if self.weight is not None:
            weight = self.weight[None, :]
            bce_loss = bce_loss * weight

        # Apply the focal loss adjustment
        focal_loss = focal_loss_adjustment * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss