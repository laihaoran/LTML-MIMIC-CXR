import torch
import torch.nn as nn

from ..registry import LOSSES
from .cross_entropy_loss import cross_entropy, binary_cross_entropy, partial_cross_entropy, kpos_cross_entropy
import math

@LOSSES.register_module
class NoiseFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_kpos=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 gamma=2,
                 balance_param=0.25,
                 LL_param =dict(mod_scheme='LL-Ct',
                 clean_rate=1,
                 delta_rel=0.1),
                 ):
        super(NoiseFocalLoss, self).__init__()
        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.use_kpos = use_kpos
        self.partial = partial
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.LL_param = LL_param
        self.gamma = gamma
        self.balance_param = balance_param

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

        logpt = - self.cls_criterion(cls_score, label, weight, reduction='none',
                                        avg_factor=avg_factor)
        # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        loss = self.loss_weight * balanced_focal_loss
        return loss
