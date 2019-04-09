from __future__ import print_function, absolute_import

import os

from reid import datasets
from reid import models

name = 'DukeMTMC_reID'
root = os.getcwd()
combine_trainval = True

dataset = datasets.create('dukemtmc_reID', os.path.join(root, 'data', name))

num_classes = (dataset.num_trainval_ids if combine_trainval else dataset.num_train_ids)

model = models.create('resnet50', num_classes=num_classes)