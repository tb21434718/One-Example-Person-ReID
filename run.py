from __future__ import print_function, absolute_import

import numpy as np
import os
import torch

from torch import nn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.exclusive_loss import Exclusive_Loss
from reid.evaluators import extract_features
from reid.trainers import Trainer
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint, save_checkpoint


##########################################################################################
# Load dataset.
root = os.getcwd()
dataset_name = 'dukemtmc_reID'
labeled_data, unlabeled_data = [], []
#dataset_name, model_name = 'dukemtmc_reID', 'resnet50'
dataset = datasets.create(dataset_name, os.path.join(root, 'data', dataset_name))
##########################################################################################
# Get the labeled_data and unlabeled_data.
import pickle
import random
seed, num, pids = 0, 1, {}
random.seed(seed); np.random.seed(seed)
fpath = './examples/{}/{}_init_{}.pickle'.format(dataset_name, dataset_name, seed)
if os.path.exists(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
        labeled_data = data['labeled']
        unlabeled_data = data['unlabeled']
    print('Load labeled_data and unlabeled_data from {}.'.format(fpath))
else:
    if num == 'all':
        labeled_data = dataset.train
    else:
        for (fname, pid, cam, index) in dataset.train:
            if pid not in pids:
                pids[pid] = []
            pids[pid].append((fname, pid, cam, index))
        for pid in pids:
            for _ in range(num):
                data = random.choice(pids[pid])
                labeled_data.append(data)
                pids[pid].remove(data)
            unlabeled_data.extend(pids[pid])
    with open(fpath, 'wb') as f:
        pickle.dump({'labeled' : labeled_data, 'unlabeled' : unlabeled_data}, f)
##########################################################################################
def get_loader(data, root, height=256, width=128, batch_size=32, workers=0, training=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if training:
        transformer = T.Compose([
            T.RandomSizedRectCrop(height, width), # 对图像进行随机裁剪并缩放.
            T.RandomHorizontalFlip(), # 对给定的PIL.Image进行随机水平翻转,概率为0.5,属于数据增强.
            T.ToTensor(),# 将numpy图像转换为torch图像.
            normalizer,
        ])
    else:
        transformer = T.Compose([
            T.RectScale(height, width), # 缩放图像.
            T.ToTensor(),
            normalizer,
        ])
        batch_size = batch_size * 8
    data_loader = DataLoader(
        Preprocessor(data, root=root, transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=training, pin_memory=True)
    return data_loader
##########################################################################################
LR, epochs = 0.1, 70
#
model_name = 'eug'
model = models.create(model_name, dropout=0.5, num_classes=dataset.num_train_ids)
#
start_step, total_step, resume = 0, 10, False
fpath = os.path.join(root, 'logs', dataset_name, model_name)
#
if resume:
    checkpoint = load_checkpoint(fpath)
    if checkpoint:
        start_step = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
#
model = nn.DataParallel(model).cuda()
#
labeled_set, unlabeled_set = labeled_data, unlabeled_data
#
for step in range(start_step, total_step):
    #
    ratio = (step + 1) * 8 / 100
    num_to_select = int(len(unlabeled_data) * ratio)
    print('Running...\n Step {}\t Ratio {}\t Num_to_select {}'.format(step, ratio, num_to_select))
    #
    # The base parameters for the backbone (e.g. ResNet50)
    base_param_ids = set(map(id, model.module.CNN.base.parameters()))
    base_params_requires_grad = filter(lambda p : p.requires_grad, model.module.CNN.base.parameters())
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    # set the learning rate for backbone to be 0.1 times
    param_groups = [
        {'params': base_params_requires_grad, 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]
    # Optimizer.
    optimizer = torch.optim.Adam(param_groups, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    # Change the learning rate by step.
    def adjust_lr(epoch, step_size=55):
        lr = 1e-3 * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1) # !!! !!!
        return epoch < step_size # Use unselect data?
    #
    labeled_loader = get_loader(labeled_set, dataset.images_dir, training=True)
    unlabeled_loader = get_loader(unlabeled_set, dataset.images_dir, training=True)
    #
    criterion = Exclusive_Loss(len(unlabeled_set), t=10).cuda()
    #
    trainer = Trainer(model, unlabeled_criterion=criterion)
    #
    for epoch in range(epochs):
        #
        trainer.train(epoch, labeled_loader, unlabeled_loader, optimizer, use_unselect_data=adjust_lr(epoch, step_size=55))
    save_checkpoint({
        'step': step,
        'state_dict': model.state_dict()
    }, fpath, 'checkpoint_' + str(step) + '.pt')
    # Get the labeled__features and unlabeled_features.
    labeled__features, _ = extract_features(model, get_loader(labeled_data, dataset.images_dir, training=False))
    unlabeled_features, _ = extract_features(model, get_loader(unlabeled_data, dataset.images_dir, training=False))
    # Calculate the distance between labeled__features and unlabeled_features.
    x = torch.cat([unlabeled_features[f].unsqueeze(0) for f, _, _, _ in unlabeled_data], 0)
    y = torch.cat([labeled__features[f].unsqueeze(0) for f, _, _, _ in labeled_data], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    #
    assert len(unlabeled_data) == dist.size(0)
    #
    prseudo_labels, prseudo_confidence = np.zeros(len(unlabeled_data)), np.zeros(len(unlabeled_data))
    values, indexs = torch.min(dist, 1) # 这里的values与diff并不相同,但是indexs与index_min等价.
    for index in range(len(unlabeled_data)):
        prseudo_confidence[index], prseudo_labels[index] = values[index], indexs[index]
    # Select some data according to the prseudo_confidence.
    indexs = np.argsort(prseudo_confidence)
    #
    v = np.zeros(dist.size(0))
    for i in range(num_to_select):
        v[indexs[i]] = 1
    v = v.astype('bool')
    # Generate the selected_data and unselected_data.
    correct, total = 0, 0
    selected_data, unselected_data = [], []
    for i, select in enumerate(v):
        if select:
            selected_data.append((unlabeled_data[i][0], prseudo_labels[i], unlabeled_data[i][2], unlabeled_data[i][3]))
            total += 1
            if unlabeled_data[i][1] == int(prseudo_labels[i]):
                correct += 1
        else:
            unselected_data.append(unlabeled_data[i])
    print('A total of {} data were selected, of which {} data were predicted correctly and the accuracy was {}'.format(total, correct, correct / total))
    # Generate the new train_data and non_train_data.
    labeled_set = labeled_data + selected_data
    unlabeled_set = unselected_data
    #evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
