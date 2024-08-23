import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

from BaseClass import _AbstractDiceLoss
from base import *
import numpy as np



__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'DiceFocalLoss', 'Generalized_dice_loss', 'DiceLoss']

class DiceLoss(_AbstractDiceLoss):
    def __init__(self, classes=44, skip_index_after=None, weight=None, sigmoid_normalization=True ):
        super().__init__(weight, sigmoid_normalization)
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        # np.save(r'D:\project\code\medicalzoopytorch-master\target.npy', target.detach().numpy())
        return compute_per_channel_dice(input, target, weight=self.weight)

class BCEDiceLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.class_nums = config['num_classes']

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)

        num = target.size(0)
        input = input.view(num, 1, -1)
        target = target.view(num, 1, -1)

        intersection = (input * target)
        dice = (2. * intersection.sum(2) + smooth) / (input.sum(2) + target.sum(2) + smooth)
        # dice = dice.sum(1) / self.class_nums
        dice = 1 - dice.sum(1) / 1

        return 0.5 * bce + dice.mean()


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + self.smooth) / (input.sum(1) + target.sum(1) + self.smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smooth=1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input, target):
        input_prob = torch.sigmoid(input)
        focal_weight = self.alpha * target * torch.pow(1 - input_prob, self.gamma)
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        focal_bce_loss = focal_weight * bce_loss
        focal_loss = focal_bce_loss.mean()
        return focal_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, alpha=0.25, gamma=2, smooth=1e-5):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, smooth=smooth)

    def forward(self, input, target):
        dice_loss = self.dice_loss(input, target)
        focal_loss = self.focal_loss(input, target)
        loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * focal_loss
        return loss


class Generalized_dice_loss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = 1e-5

    def forward(self, input, target):
        wei = torch.sum(target, axis=[0,2,3])
        wei = 1/(wei**2 + self.smooth)
        intersection = torch.sum(wei*torch.sum(input * target, axis=[0,2,3]))
        union = torch.sum(wei*torch.sum(input + target, axis=[0,2,3]))
        gldice_loss = 1- (2. * intersection)/(union+self.smooth)
        return gldice_loss
