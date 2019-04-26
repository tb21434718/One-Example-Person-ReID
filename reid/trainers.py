# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:57:21 2019

@author: chen
"""
from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class BaseTrainer(object):
    def __init__(self, model, lamda=0.8, labeled_criterion=torch.nn.CrossEntropyLoss().cuda(), unlabeled_criterion=torch.nn.CrossEntropyLoss().cuda()):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.lamda = lamda
        self.labeled_criterion = labeled_criterion
        self.unlabeled_criterion = unlabeled_criterion
        

    def train(self, epoch, labeled_dataloader, unlabeled_dataloader, optimizer, use_unselect_data, print_freq=5):
        self.model.train()
        #
        batch_time, loss, CE_loss, Exclusive_loss, CE_prec, Exclusive_prec = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        
        end = time.time()
        
        unlabeled_dataloader = iter(cycle(unlabeled_dataloader)) # !!! 什么意思呢??? !!!
        
        for i, inputs in enumerate(labeled_dataloader):
            #
            labeled_inputs, labeled_targets = self._parse_data(inputs, 'labeled')
            CE_loss, CE_prec = self._forward(labeled_inputs, labeled_targets, 'labeled')
            #
            Exclusive_loss = 0
            if use_unselect_data:
                unlabeled_inputs = next(unlabeled_dataloader)
                unlabeled_inputs, unlabeled_targets = self._parse_data(unlabeled_inputs, 'unlabeled')
                Exclusive_loss, Exclusive_prec = self._forward(unlabeled_inputs, unlabeled_targets, 'unlabeled')
            #
            loss = CE_loss * self.lamda + Exclusive_loss * (1 - self.lamda)
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            loss.update(loss.item(), labeled_targets.size(0))
            CE_loss.update(CE_loss.item(), labeled_targets.size(0))
            Exclusive_loss.update(Exclusive_loss.item())
            CE_prec.update(CE_prec, labeled_targets.size(0))
            Exclusive_prec.update(Exclusive_prec, labeled_targets.size(0))
            #
            batch_time.update(time.time() - end)
            end = time.time()
            #
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'CE_Loss {:.3f} ({:.3f})\t'
                      'Exclusive_Loss {:.3f} ({:.3f})\t'
                      'CE_Prec {:.1f} ({:.1f})\t'
                      'Exclusive_Prec {:.1%} ({:.1%})\t'
                      .format(epoch, i + 1, len(labeled_dataloader),
                              batch_time.val, batch_time.avg,
                              loss.val, loss.avg,
                              CE_loss.val, CE_loss.avg,
                              Exclusive_loss.val, Exclusive_loss.avg,
                              CE_prec.val, CE_prec.avg,
                              Exclusive_prec.val, Exclusive_prec.avg))
    
    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

class Trainer(BaseTrainer):
    def _parse_data(self, inputs, mode):
        # The format of input: "img, fname, pid, camid".
        imgs, _, pids, _, indexs = inputs
        inputs = Variable(imgs)
        if mode == 'labeled':
            targets = Variable(pids.cuda())
        elif mode == 'unlabeled':
            targets = Variable(indexs.cuda()) # !!! 这里为什么要加.cuda()??? !!!
        else:
            raise KeyError
        return inputs, targets

    def _forward(self, inputs, targets, mode):
        labeled_prediction, unlabeled_feature = self.model(inputs)
        if mode == 'labeled':
            CE_loss = self.labeled_criterion(labeled_prediction, targets)
            CE_prec, = accuracy(labeled_prediction.data, targets.data)
            CE_prec = CE_prec[0]
            return CE_loss, CE_prec
        elif mode == 'unlabeled':
            Exclusive_loss, outputs = self.unlabeled_criterion(unlabeled_feature, targets)
            Exclusive_prec = accuracy(outputs.data, targets.data)
            Exclusive_prec = Exclusive_prec[0]
            return Exclusive_loss, Exclusive_prec
        else:
            raise KeyError
        '''
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
        '''