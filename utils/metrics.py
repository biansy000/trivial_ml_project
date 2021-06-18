import torch
import numpy as np
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_IoU(pred, gt):
    # calculate intersection over union
    size = pred.size

    intersect = np.logical_and((pred > 0.5), (gt > 0.5))
    union = np.logical_or((pred > 0.5), (gt > 0.5))

    return (intersect.sum() / union.sum()), size


def calc_Acc(pred, gt):
    # calculate accuracy
    size = pred.size

    true_pos = np.logical_and((pred > 0.5), (gt > 0.5))
    true_neg = np.logical_and((pred <= 0.5), (gt <= 0.5))

    acc = (true_pos.sum() + true_neg.sum()) / size

    return acc, size


def weighted_BCE_loss(input, target, positive_weight=2): 
    # weighted BCE loss, positive samples can be given larger weights for more stable training
    input = torch.clamp(input, min=1e-12, max=1-1e-12)
    target = target.expand(input.shape)

    weights = torch.ones_like(target)
    weights[target > 0.5] = positive_weight

    out = nn.functional.binary_cross_entropy(input, target, reduction='none')
    # assert out.shape == weights.shape, f'{out.shape}, {weights.shape}'
    out = out * weights
    
    return out.mean()