from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..registry import LOSSES
import numpy as np
import math

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
class NoiseCrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_kpos=False,
                 partial=False,
                 reduction='mean',
                 loss_weight=1.0,
                LL_param =dict(mod_scheme='LL-Ct',
                 clean_rate=1,
                 delta_rel=0.1),
                 thrds=None):
        super(NoiseCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.use_kpos = use_kpos
        self.partial = partial
        self.reduction = reduction
        self.LL_param = LL_param
        self.loss_weight = loss_weight
        if self.use_sigmoid and thrds is not None:
            self.thrds=inverse_sigmoid(thrds)
        else:
            self.thrds = thrds

        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        elif self.use_kpos:
            self.cls_criterion = kpos_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def number(self, epoch):
        if epoch == 1:
            self.LL_param['clean_rate'] = 0.9
        elif epoch == 2:
            self.LL_param['clean_rate'] = 0.8
        elif epoch == 3:
            self.LL_param['clean_rate'] = 0.7
        elif epoch == 4:
            self.LL_param['clean_rate'] = 0.6
        elif epoch == 5:
            self.LL_param['clean_rate'] = 0.5
        elif epoch == 6:
            self.LL_param['clean_rate'] = 0.4
        elif epoch == 7:
            self.LL_param['clean_rate'] = 0.3
        elif epoch == 8:
            self.LL_param['clean_rate'] = 0.2


    def forward(self,
                cls_score,
                label,
                epoch=1,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
       
        assert cls_score.dim() == 2
        
        batch_size = int(cls_score.size(0))
        num_classes = int(cls_score.size(1))
        
        unobserved_mask = (label == 0)  ## 找到所有0类，作为mask
        
        # compute loss for each image and class:
        loss_matrix, corrected_loss_matrix = self.loss_an(cls_score, label.clip(0), weight, avg_factor, reduction_override)

        correction_idx = None
        print('This epoch is {}'.format(epoch))
        self.number(epoch)
        if epoch == 0: # if epoch is 1, do not modify losses
            final_loss_matrix = loss_matrix
        else:
            if self.LL_param['mod_scheme'] == 'LL-Cp':  # 永久修改标签
                k = math.ceil(batch_size * num_classes * self.LL_param['delta_rel'])
            else:
                k = math.ceil(batch_size * num_classes * (1-self.LL_param['clean_rate']))
        
            unobserved_loss = unobserved_mask.bool() * loss_matrix  ## 取出0类损失
            topk = torch.topk(unobserved_loss.flatten(), k) ## 取前k个损失值
            topk_lossvalue = topk.values[-1]  ##取topk的下界
            correction_idx = torch.where(unobserved_loss > topk_lossvalue)  ## 找到过大的值
            if self.LL_param['mod_scheme'] in ['LL-Ct', 'LL-Cp']:
                final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)  ## 如果loss大于阈值，认为该标签出错，按照反方向重新计算损失。
            else:
                zero_loss_matrix = torch.zeros_like(loss_matrix)
                final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)
                ## 如果loss大于阈值，认为该标签出错，不计算该损失。

        main_loss = final_loss_matrix.mean()
        
        # return main_loss, correction_idx
        return main_loss

    def loss_an(self, cls_score,
                label,
                weight,
                avg_factor,
                reduction_override):
        # assert torch.min(observed_labels) >= 0  ## 必须包含正类， 可不需要
        # compute loss:
        # loss_matrix = F.binary_cross_entropy_with_logits(cls_score, label, reduction='none')
        # corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(observed_labels).float(), reduction='none')  # 吧所以0类都赋值为1，计算损失
        loss_matrix = self.loss_an_an(cls_score=cls_score,
                label=label,
                weight=weight,
                avg_factor=avg_factor,
                reduction_override=reduction_override)

        corrected_loss_matrix = self.loss_an_an(cls_score=cls_score,
                label=torch.logical_not(label),
                weight=weight,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
        return loss_matrix, corrected_loss_matrix

    
    def loss_an_an(self, cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.thrds is not None:
            cut_high_mask = (label == 1) * (cls_score > self.thrds[1])
            cut_low_mask = (label == 0) * (cls_score < self.thrds[0])
            if weight is not None:
                weight *= (1 - cut_high_mask).float() * (1 - cut_low_mask).float()
            else:
                weight = (1 - cut_high_mask).float() * (1 - cut_low_mask).float()

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            )
        return loss_cls



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

