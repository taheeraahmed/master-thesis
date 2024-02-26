import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """
    Implmenetation of the weighted focal loss function taken from with minor modifications:
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

    :param alpha: A tensor of shape [num_classes] containing weights for each class.
    :param gamma: Focusing parameter to adjust the rate at which easy examples are down-weighted.
    :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.
        :param inputs: Predictions from the model (logits, before softmax).
        :param targets: True labels.
        :return: Weighted Focal Loss value.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) 
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            F_loss = at * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss