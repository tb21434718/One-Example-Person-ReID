# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 10:49:29 2019

@author: chen
"""
from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .resnet import *

__all__ = ['EUG']

class AvgPool(nn.Module):
    def __init__(self, dropout, num_classes, input_feature_size=2048):
        super(AvgPool, self).__init__()
        
        CE_feature_size = 1024
        Exclusive_feature_size = 2048
        
        self.CE_embeding = nn.Linear(input_feature_size, CE_feature_size)
        self.CE_embeding_bn = nn.BatchNorm1d(CE_feature_size) # 对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作.
        
        self.Exclusive_embeding = nn.Linear(input_feature_size, Exclusive_feature_size)
        self.Exclusive_embeding_bn = nn.BatchNorm1d(Exclusive_feature_size)
        
        init.kaiming_normal_(self.CE_embeding.weight, mode='fan_out')
        init.constant_(self.CE_embeding.bias, 0)
        
        init.constant_(self.CE_embeding_bn.weight, 1)
        init.constant_(self.CE_embeding_bn.bias, 0)

        init.kaiming_normal_(self.Exclusive_embeding.weight, mode='fan_out')
        init.constant_(self.Exclusive_embeding.bias, 0)
        
        init.constant_(self.Exclusive_embeding_bn.weight, 1)        
        init.constant_(self.Exclusive_embeding_bn.bias, 0)
        
        self.drop = nn.Dropout(dropout)
        
        self.classify_fc = nn.Linear(CE_feature_size, num_classes, bias=True)
        init.normal_(self.classify_fc.weight, std = 0.001)
        init.constant_(self.classify_fc.bias, 0)
    
    def forward(self, inputs):
        pool5 = inputs.mean(dim=1)
        if (not self.training):
            return F.normalize(pool5, p=2, dim=1)
        ''' CE '''
        net = self.drop(pool5)
        net = self.CE_embeding(net)
        net = self.CE_embeding_bn(net)
        net = F.relu(net)
        net = self.drop(net)
        labeled_prediction = self.classify_fc(net)
        ''' Exclusive '''
        net = self.drop(pool5)
        net = self.Exclusive_embeding(net)
        net = self.Exclusive_embeding_bn(net)
        net = F.normalize(net, p=2, dim=1) # L2标准化.
        Exclusive_feature = self.drop(net)
        #
        return labeled_prediction, Exclusive_feature
        

class EUG(nn.Module):
    def __init__(self, dropout=0.5, num_classes=702, fixed_layer=False):
        super(self.__class__, self).__init__()
        self.CNN = resnet50(dropout=dropout, fixed_layer=fixed_layer)
        self.avg_pool = AvgPool(dropout, num_classes)
     
    def forward(self, x):
        resnet_features = self.CNN(x)
        labeled_prediction, Exclusive_feature = self.avg_pooling(resnet_features)
        return labeled_prediction, Exclusive_feature