import torch
from models import ModelConfig
import torch
import torch.nn as nn

nINF = -100


class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp +
                                       pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp +
                                        pmask, dim=1).mul(self.Tp)[sample_mask]

        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(
            x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(
            x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
            torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        """
        https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
        """
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class FocalLoss(nn.Module):
    """
    Taken from: 

    """

    def __init__(self, alpha=0.25, gamma=2.5, weights=None, device='cuda', reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.reduction = reduction

        if weights is not None:
            weights = weights.to(device)

        self.loss = nn.BCEWithLogitsLoss(weight=weights)
        self.to(device)

    def forward(self, inputs, targets):
        '''
        Apply focal loss for multi-label classification.
        :param inputs: Tensor of size (batch_size, num_classes) representing logit outputs
        :param targets: Tensor of size (batch_size, num_classes) representing true labels
        :return: Computed focal loss
        '''
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        bce_loss = self.loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


def set_criterion(model_config: ModelConfig, class_weights: torch.Tensor = None) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_config.loss_arg == 'mlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    elif model_config.loss_arg == 'wmlsm':
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    elif model_config.loss_arg == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif model_config.loss_arg == 'wbce':
        criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    elif model_config.loss_arg == 'focal':
        criterion = FocalLoss(device=device, gamma=2.0)
    elif model_config.loss_arg == 'wfocal':
        criterion = FocalLoss(device=device, gamma=2.0, weights=class_weights)
    elif model_config.loss_arg == 'asl':
        criterion = AsymmetricLoss()
    elif model_config.loss_arg == 'twoway':
        criterion = TwoWayLoss()
    else:
        raise ValueError(f'Invalid loss argument: {model_config.loss_arg}')
    return criterion
