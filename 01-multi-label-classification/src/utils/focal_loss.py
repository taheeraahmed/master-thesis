import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class WeightedFocalLoss(nn.Module):
    """
    Implmenetation of the weighted focal loss function taken from with minor modifications:
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
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

    def forward(self, input, target):
        """
        Apply the focal loss between input and target

        Args:
            input: Tensor of shape (N, C) where N is the batch size and C is the number of classes.
            target: Tensor of shape (N,) where each value is 0 ≤ targets[i] ≤ C−1.
        """
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.long()  # Ensures target is torch.int64
        #target = target.view(-1, 1) TODO: For some reason this line is causing an error
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
