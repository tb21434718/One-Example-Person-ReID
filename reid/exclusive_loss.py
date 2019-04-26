# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:43:41 2019

@author: chen
"""
from __future__ import absolute_import

import torch
import torch.nn.functional as F

from torch import nn, autograd


class Exclusive(autograd.Function):
    def __init__(self, V):
        super(Exclusive, self).__init__()
        self.V = V # V用于存储每个序号标签数据的特征.
    
    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.V.t())
        return outputs
    
    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = grad_outputs.mm(self.V) if self.needs_input_grad[0] else None
        for x, y in zip(inputs, targets):
            self.V[y] = F.normalize( (self.V[y] + x) / 2, p=2, dim=0) # 对V进行更新并归一化.
        return grad_inputs, None


class Exclusive_Loss(nn.Module):
    def __init__(self, num_classes, num_features=1024, t=1.0, weight=None):
        super(Exclusive_Loss,self).__init__()
        self.num_features = num_features
        self.t = t
        self.weight = weight
        self.register_buffer('V', torch.zeros(num_classes, num_features))
    
    def forward(self, inputs, targets):
        outputs = Exclusive(self.V)(inputs, targets) * self.t
        Exclusive_loss = F.cross_entropy(outputs, targets, weight=self.weight)
        return Exclusive_loss, outputs