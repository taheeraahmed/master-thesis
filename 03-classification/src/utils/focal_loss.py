import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
        """
        Focal loss for imbalanced datasets.
        :param class_weights: (alpha) Weights for each class. If None, it will be calculated.
        :param gamma: Focusing parameter.
        :param reduction: Reduction method.
        """
        super(FocalLoss, self).__init__()
        self.class_weights = class_weights  # Class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Apply focal loss.
        :param inputs: Predictions from the model (logits before softmax).
        :param targets: True class labels.
        :return: Computed focal loss.
        """
        # Convert inputs to probabilities
        probs = torch.softmax(inputs, dim=1)
        # Gather the probabilities of the true classes for each sample
        target_probs = probs.gather(dim=1, index=targets.view(-1, 1)).view(-1)
        # Compute the focal loss
        focal_loss = -self.class_weights[targets] * \
            ((1 - target_probs) ** self.gamma) * target_probs.log()

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Example usage
# Initialize FocalLoss with class weights
# focal_loss = FocalLoss(weight=pos_weights_tensor, gamma=2.0)

# Compute loss
# loss = focal_loss(predictions, targets)
