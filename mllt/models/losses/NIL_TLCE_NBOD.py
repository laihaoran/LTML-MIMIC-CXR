from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..registry import LOSSES
import numpy as np


def taylor_crossentropy(pred, target, n=2, alpha=0.1, beta=1.0, epsilon=1e-6):
    """Calculate Taylor-Cross Entropy Loss for multi-label classification tasks.

    Args:
        pred (torch.Tensor): Predicted probability. The shape of the tensor should be (N, C),
            where N is the batch size, and C is the number of classes.
        target (torch.Tensor): Ground-truth label. The shape of the tensor should be (N, C).
        n (int): Order of the approximation. Default: 2.
        alpha (float): Weighting factor for the diagonal terms. Default: 0.1.
        beta (float): Weighting factor for the non-diagonal terms. Default: 1.0.
        epsilon (float): A small value to avoid numerical instability. Default: 1e-6.

    Returns:
        torch.Tensor: The calculated Taylor-Cross Entropy Loss.
    """

    ## with sigmoid

    pred = F.sigmoid(pred)
    # calculate the diagonal terms
    diag_terms = torch.pow(pred, n) * torch.log(pred + epsilon) * target
    diag_terms = torch.sum(diag_terms, dim=1)

    # calculate the non-diagonal terms
    non_diag_terms = (1 - pred) * target
    non_diag_terms = non_diag_terms.unsqueeze(2) * non_diag_terms.unsqueeze(1)
    non_diag_terms = beta * torch.sum(torch.triu(non_diag_terms, diagonal=1), dim=(1, 2))

    # combine the diagonal and non-diagonal terms
    loss = alpha * diag_terms - non_diag_terms
    return -torch.mean(loss)



def _squeeze_binary_labels(label):
    if label.size(1) == 1:
        squeeze_label = label.view(len(label), -1)
    else:
        inds = torch.nonzero(label >= 1).squeeze()
        squeeze_label = inds[:,-1]
    return squeeze_label

def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    if label.size(-1) != pred.size(0):
        label = _squeeze_binary_labels(label)

    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def partial_cross_entropy(pred,
                          label,
                          weight=None,
                          reduction='mean',
                          avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    mask = label == -1
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    if mask.sum() > 0:
        loss *= (1-mask).float()
        avg_factor = (1-mask).float().sum()

    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

def kpos_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    target = label.float() / torch.sum(label, dim=1, keepdim=True).float()

    loss = - target * F.log_softmax(pred, dim=1)
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

@LOSSES.register_module
class NBOD_TaylorCrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_kpos=False,
                 partial=False,
                 reduction='mean',
                 multi_classifier_diversity_factor=1.0,
                 multi_classifier_diversity_factor_hcm=1.0,
                 ce_ratio= 1.0,
                 hcm_ratio=1.0,
                 hcm_N = 15,
                 thrds=None):
        super(NBOD_TaylorCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.use_kpos = use_kpos
        self.partial = partial
        self.reduction = reduction
        self.hcm_N = hcm_N
        self.multi_classifier_diversity_factor = multi_classifier_diversity_factor
        self.multi_classifier_diversity_factor_hcm = multi_classifier_diversity_factor_hcm
        self.ce_ratio = ce_ratio 
        self.hcm_ratio = hcm_ratio 
        if self.use_sigmoid and thrds is not None:
            self.thrds=inverse_sigmoid(thrds)
        else:
            self.thrds = thrds

        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = taylor_crossentropy
        elif self.use_kpos:
            self.cls_criterion = kpos_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        classifier_num = len(cls_score)
        inputs_HCM_balance = []
        inputs_balance = []
        loss_HCM = 0
        loss = 0
        los_ce = 0

        # class_select = inputs[0].scatter(1, targets[0].unsqueeze(1), 999999)
        class_select = cls_score[0] + label * 999999
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_N]
        mask = torch.zeros_like(cls_score[0]).scatter(1, class_select_include_target, 1)

        ## cut over thred
        if self.thrds is not None:
            cut_high_mask = (label == 1) * (cls_score[0] > self.thrds[1])
            cut_low_mask = (label == 0) * (cls_score[0] < self.thrds[0])
            if weight is not None:
                weight *= (1 - cut_high_mask).float() * (1 - cut_low_mask).float()
            else:
                weight = (1 - cut_high_mask).float() * (1 - cut_low_mask).float()
        for i in range(classifier_num):

            inputs_balance.append(cls_score[i])
            inputs_HCM_balance.append(cls_score[i] * mask)

            los_ce += self.cls_criterion(
                cls_score[i],
                label,
                **kwargs)
            loss_HCM += self.cls_criterion(
                cls_score[i] * mask,
                label,
                **kwargs)
        loss += NBOD_ML(inputs_balance, factor=self.multi_classifier_diversity_factor)
        loss += NBOD_ML(inputs_HCM_balance, factor=self.multi_classifier_diversity_factor_hcm)

        loss += los_ce * self.ce_ratio + loss_HCM * self.hcm_ratio
        return loss

def inverse_sigmoid(Y):
    X = []
    for y in Y:
        y = max(y,1e-14)
        if y == 1:
            x = 1e10
        else:
            x = -np.log(1/y-1)
        X.append(x)

    return X



def NBOD_ML(inputs, factor):

    classifier_num = len(inputs)
    if classifier_num == 1:
        return 0
    logits_softmax = []
    logits_logsoftmax = []
    for i in range(classifier_num):
        logits_softmax.append(F.sigmoid(inputs[i]))
        logits_logsoftmax.append(torch.log(logits_softmax[i] + 1e-9))

    loss_mutual = 0
    for i in range(classifier_num):
        for j in range(classifier_num):
            if i == j:
                continue
            loss_mutual += factor * F.kl_div(logits_logsoftmax[i], logits_softmax[j],reduction='batchmean')
    loss_mutual /= (classifier_num - 1)
    return  loss_mutual

if __name__ == '__main__':
    pass

