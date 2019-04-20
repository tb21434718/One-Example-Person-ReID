from __future__ import print_function, absolute_import

import numpy as np
import os
import torch

#from collections import OrderedDict
#from sklearn.metrics import average_precision_score
from torch import nn
#from torch.autograd import Variable
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
#from reid.dist_metric import DistanceMetric
from reid.evaluators import extract_features, pairwise_distance, Evaluator
from reid.trainers import Trainer
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint, save_checkpoint


##########################################################################################
# Load dataset.
import pickle
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
root = os.getcwd()
num, pids =  1, {}
labeled_data, unlabeled_data = [], []
dataset_name, model_name = 'dukemtmc_reID', 'resnet50'
dataset = datasets.create(dataset_name, os.path.join(root, 'data', dataset_name))
fpath = './examples/{}/{}_init_{}.pickle'.format(dataset_name, dataset_name, seed)
if os.path.exists(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
        labeled_data = data['labeled']
        unlabeled_data = data['unlabeled']
    print('Load one-shot split from {}.'.format(fpath))
else:
    if num == 'all':
        labeled_data = dataset.trainval
    else:
        for (fname, pid, cam) in dataset.trainval:
            if pid not in pids:
                pids[pid] = []
            pids[pid].append((fname, pid, cam))
        import random
        for pid in pids:
            for _ in range(num):
                data = random.choice(pids[pid])
                labeled_data.append(data)
                pids[pid].remove(data)
            unlabeled_data.extend(pids[pid])
    with open(fpath, 'wb') as f:
        pickle.dump({'labeled':labeled_data, 'unlabeled':unlabeled_data}, f)

##########################################################################################
def get_loader(data, root, height, width, batch_size=16, workers=0, training=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if training:
        transformer = T.Compose([
            T.RandomSizedRectCrop(height, width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ])
    else:
        transformer = T.Compose([
            T.RectScale(height, width),
            T.ToTensor(),
            normalizer,
        ])
    data_loader = DataLoader(
        Preprocessor(data, root=root, transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=training, pin_memory=True)
    return data_loader
#train_loader = get_loader(labeled_data, dataset.images_dir, 256, 128, batch_size=4, workers=0, training=True)
##########################################################################################
num_classes = dataset.num_trainval_ids
model = models.create(model_name, num_classes=num_classes)

logs_dir = os.path.join(root, 'logs')
start_step, total_step, resume = 0, 10, False
if resume:
    fpath = os.path.join(logs_dir, dataset_name, model.name)
    checkpoint = load_checkpoint(fpath)
    if checkpoint:
        start_step = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
model = nn.DataParallel(model).cuda()
# Optimizer.
if hasattr(model.module, 'base'):
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]
else:
    param_groups = model.parameters()

optimizer = torch.optim.Adam(param_groups, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

# Evaluator.
evaluator = Evaluator(model)

# Criterion.
criterion = nn.CrossEntropyLoss().cuda()

trainer = Trainer(model, criterion)

LR, epochs = 0.1, 50
# Schedule learning rate
def adjust_lr(epoch):
    step_size = 40
    lr = LR * (0.1 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)


num_to_select = 100

for step in range(start_step, total_step):
    
    train_loader = get_loader(labeled_data, dataset.images_dir, 256, 128, batch_size=4, workers=0, training=True)
    test_loader = get_loader(unlabeled_data, dataset.images_dir, 256, 128, batch_size=4, workers=0, training=True) # !!! !!!
    
    for epoch in range(epochs):
        adjust_lr(epoch)
        
        #
        trainer.train(epoch, train_loader, optimizer)
        #evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        train_features, labels = extract_features(model, train_loader)
        test_features, _ = extract_features(model, test_loader)
        
        x = torch.cat([test_features[f].unsqueeze(0) for f, _, _ in unlabeled_data], 0)
        y = torch.cat([train_features[f].unsqueeze(0) for f, _, _ in labeled_data], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        
        prseudo_labels, prseudo_confidence = np.zeros(dist.size(0)), np.zeros(dist.size(0))
        values, indexs = torch.min(dist, 1)
        for index in range(dist.size(0)):
            prseudo_confidence[index], prseudo_labels[index] = values[index], indexs[index]
        
        indexs = np.argsort(prseudo_confidence)
        
        v = np.zeros(dist.size(0))
        for i in range(num_to_select):
            v[indexs[i]] = 1
        v = v.astype('bool')
        
        
        correct, total = 0, 0
        selected_data, unselected_data = [], []
        for i, select in enumerate(v):
            if select:
                selected_data.append((unlabeled_data[i][0], prseudo_labels[i], unlabeled_data[i][2]))
                total += 1
                if unlabeled_data[i][1] == int(prseudo_labels[i]):
                    correct += 1
            else:
                unselected_data.append(unlabeled_data[i])
                
        labeled_data += selected_data
        unlabeled_data = unselected_data
        
        break
    '''
    save_checkpoint({
        'step': step,
        'state_dict': model.state_dict()
    }, fpath, 'checkpoint_' + str(step) + '.pt
    '''
    break
