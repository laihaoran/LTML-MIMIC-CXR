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


@LOSSES.register_module
class NBOD_ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 multi_classifier_diversity_factor=1.0,
                 multi_classifier_diversity_factor_hcm=1.0,
                 ce_ratio= 1.0,
                 hcm_ratio=1.0,
                 hcm_N = 15,
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
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 freq_file='./class_freq.pkl'):
        super(NBOD_ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.hcm_N = hcm_N
        self.multi_classifier_diversity_factor = multi_classifier_diversity_factor
        self.multi_classifier_diversity_factor_hcm = multi_classifier_diversity_factor_hcm
        self.ce_ratio = ce_ratio 
        self.hcm_ratio = hcm_ratio 
        self.reduction = reduction

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


        weight = self.reweight_functions(label)
        
  
        for i in range(classifier_num):
            temp_cls_score, temp_weight = self.logit_reg_functions(label.float(), cls_score[i], weight)
            inputs_balance.append(temp_cls_score)
            inputs_HCM_balance.append(temp_cls_score * mask)
            if self.focal:
                los_ce += self.fc_input(
                temp_cls_score,
                label,
                temp_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)

                loss_HCM += self.fc_input(
                temp_cls_score * mask,
                label,
                temp_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)

            else:
                los_ce += self.cls_criterion(temp_cls_score, label.float(), temp_weight,reduction=reduction)

                loss_HCM += self.cls_criterion(temp_cls_score * mask, label.float(), temp_weight,reduction=reduction)

        loss += NBOD_ML(inputs_balance, factor=self.multi_classifier_diversity_factor)
        loss += NBOD_ML(inputs_HCM_balance, factor=self.multi_classifier_diversity_factor_hcm)

        loss += los_ce * self.ce_ratio + loss_HCM * self.hcm_ratio

        return loss

    def fc_input(self, cls_score, label, weight, reduction,
                                     avg_factor):
        logpt = - self.cls_criterion(cls_score, label, None, reduction=reduction,
                                     avg_factor=avg_factor)
        # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
        pt = torch.exp(logpt)
        loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
        focal_loss = ((1 - pt) ** self.gamma) * loss
        balanced_focal_loss = self.balance_param * focal_loss
        loss =  balanced_focal_loss
        return loss.mean()

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