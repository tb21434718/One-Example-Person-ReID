from __future__ import print_function, absolute_import

import os
import torch

from torch import nn

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.evaluators import Evaluator
from reid.trainers import Trainer


root = os.getcwd()
name = 'DukeMTMC_reID'
combine_trainval = True

dataset = datasets.create('dukemtmc_reID', os.path.join(root, 'data', name))

num_classes = (dataset.num_trainval_ids if combine_trainval else dataset.num_train_ids)

model = models.create('resnet50', num_classes=num_classes)
model = nn.DataParallel(model).cuda()

metric = DistanceMetric(algorithm='euclidean')

evaluator = Evaluator(model)

criterion = nn.CrossEntropyLoss().cuda()

if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
else:
    param_groups = model.parameters()
optimizer = torch.optim.SGD(param_groups, lr=0.1,
                            momentum=0.9,
                            weight_decay=5e-4,
                            nesterov=True)

trainer = Trainer(model, criterion)
def adjust_lr(epoch):
    step_size = 40
    lr = 0.1 * (0.1 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)
epoch = 0
