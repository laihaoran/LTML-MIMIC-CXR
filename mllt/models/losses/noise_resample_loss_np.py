import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from .utils import weight_reduce_loss
from ..registry import LOSSES
from .cross_entropy_loss import cross_entropy, _expand_binary_labels, binary_cross_entropy, partial_cross_entropy
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.nn import Parameter
from sklearn.manifold import TSNE
import math

@LOSSES.register_module
class NoiseNPResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 focal=dict(
                     focal=True,
                     balance_param=2.0,
                     gamma=2,
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 LL_param =dict(mod_scheme='LL-Ct',
                 clean_rate=1,
                 delta_rel=0.1),

                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 freq_file='./class_freq.pkl'):
        super(NoiseNPResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.LL_param = LL_param
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.balance_param = focal['balance_param']

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = torch.from_numpy(np.asarray(
            mmcv.load(freq_file)['class_freq'])).float().cuda()
        self.neg_class_freq = torch.from_numpy(
            np.asarray(mmcv.load(freq_file)['neg_class_freq'])).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

        # print('\033[1;35m loading from {} | {} | {} | s\033[0;0m'.format(freq_file, reweight_func, logit_reg))
        # print('\033[1;35m rebalance reweighting mapping params: {:.2f} | {:.2f} | {:.2f} \033[0;0m'.format(self.map_alpha, self.map_beta, self.map_gamma))
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

        # correction_idx = None
        # print('This epoch is {}'.format(epoch))
        self.number(epoch)
        if epoch == 0: # if epoch is 1, do not modify losses
            final_loss_matrix = loss_matrix
        else:
            if self.LL_param['mod_scheme'] == 'LL-Cp':  # 永久修改标签
                k = math.ceil(batch_size * num_classes * self.LL_param['delta_rel'])
            else:
                k = math.ceil(batch_size * num_classes * (1-self.LL_param['clean_rate']))

            pos_number = torch.sum(label).item()
            p_k = math.ceil (k * 0.1)
            if p_k > pos_number:
                p_k = pos_number
            n_k = k - p_k

            unobserved_loss_0 = unobserved_mask.bool() * loss_matrix  ## 取出0类损失
            unobserved_loss_1 = label.bool() * loss_matrix ## 取出1类损失
            topk_n = torch.topk(unobserved_loss_0.flatten(), n_k) ## 取前k个损失值
            topk_p = torch.topk(unobserved_loss_1.flatten(), p_k) ## 取前k个损失值

            topk_n_lossvalue = topk_n.values[-1]  ##取topk_n的下界
            topk_p_lossvalue = topk_p.values[-1]  ##取topk_p的下界

            correction_idx_0 = torch.where(unobserved_loss_0 > topk_n_lossvalue)  ## 找到 topn 过大的值
            correction_idx_1 = torch.where(unobserved_loss_1 > topk_p_lossvalue)  ## 找到 topp 过大的值

            if self.LL_param['mod_scheme'] in ['LL-Ct', 'LL-Cp']:
                final_loss_matrix = torch.where(unobserved_loss_0 < topk_n_lossvalue, loss_matrix, corrected_loss_matrix)  ## 如果loss大于阈值，认为该标签出错，按照反方向重新计算损失。

                final_loss_matrix = torch.where(unobserved_loss_1 < topk_p_lossvalue, final_loss_matrix, corrected_loss_matrix)  ## 如果loss大于阈值，认为该标签出错，按照反方向重新计算损失。
            else:
                zero_loss_matrix = torch.zeros_like(loss_matrix)
                final_loss_matrix = torch.where(unobserved_loss_0 < topk_n_lossvalue, loss_matrix, zero_loss_matrix)
                ## 如果loss大于阈值，认为该标签出错，不计算该损失。

                final_loss_matrix = torch.where(unobserved_loss_1 < topk_p_lossvalue, final_loss_matrix, zero_loss_matrix)

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
        reduction = (reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = - self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(logpt)
            loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            loss = ((1 - pt) ** self.gamma) * loss
            loss = self.balance_param * loss
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss



    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight
