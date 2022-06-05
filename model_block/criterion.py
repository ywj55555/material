# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, model_num_outputs=3, loss_balance_weights=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )
        self.model_num_outputs = model_num_outputs
        self.loss_balance_weights = loss_balance_weights

    def _forward(self, score, target):
        # print("score:",score.size(),"target:",target.size())
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='nearest')
        # print("score:",score.size())

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):
        # print("score size:...",len(score))

        if self.model_num_outputs == 1:
            score = [score]

        weights = self.loss_balance_weights
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None, model_num_outputs=3, loss_balance_weights=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )
        self.model_num_outputs = model_num_outputs
        self.loss_balance_weights = loss_balance_weights

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='nearest')

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='nearest')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label
        # 下面这些主要是为了计算阈值 threshold
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        # min_value = pred[min(self.min_kept, pred.numel() - 1)]
        min_value = pred[min(self.min_kept * score.size(0), pred.numel() - 1)]  # 排序取前 至少min_kept 个 get 获取张量元素个数
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if self.model_num_outputs  == 1:
            score = [score]

        weights = self.loss_balance_weights
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
       # print("loss weight : ",weights, len(score), functions)
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])
