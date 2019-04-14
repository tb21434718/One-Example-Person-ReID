from __future__ import print_function, absolute_import

import numpy as np
import os
import torch

from collections import OrderedDict
from sklearn.metrics import average_precision_score
from torch import nn
#from torch.autograd import Variable
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
#from reid.dist_metric import DistanceMetric
#from reid.evaluators import Evaluator
#from reid.trainers import Trainer
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor


seed = 1
workers = 0
batch_size = 4
data_width = 128
data_height = 256
root = os.getcwd()
name = 'DukeMTMC_reID'
combine_trainval = False

np.random.seed(seed)
torch.manual_seed(seed)

dataset = datasets.create('dukemtmc_reID', os.path.join(root, 'data', name))

num_classes = (dataset.num_trainval_ids if combine_trainval else dataset.num_train_ids)

model = models.create('resnet50', num_classes=num_classes)
model = nn.DataParallel(model).cuda()

#metric = DistanceMetric(algorithm='euclidean')

#evaluator = Evaluator(model)

criterion = nn.CrossEntropyLoss().cuda()

#trainer = Trainer(model, criterion)
'''
def adjust_lr(epoch):
    step_size = 40
    lr = 0.1 * (0.1 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)
'''
'''
epoch = 0
test_transformer = T.Compose([
    T.RectScale(data_height, data_width),
    T.ToTensor(),
    normalizer,
])
'''
# Get 'train_loader'.
normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
train_transformer = T.Compose([
    T.RandomSizedRectCrop(data_height, data_width),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalizer,
])
train_loader = DataLoader(
    Preprocessor(dataset.train, root=dataset.images_dir,
                 transform=train_transformer),
    batch_size=batch_size, num_workers=workers,
    shuffle=True, pin_memory=True, drop_last=True)
# Get 'optimizer'.
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
'''
#trainer.train(epoch, train_loader, optimizer)
model.train()
for i, inputs in enumerate(train_loader):
    #inputs, targets = self._parse_data(inputs)
    imgs, _, pids, _ = inputs
    inputs = Variable(imgs)
    targets = Variable(pids.cuda())
    #loss, prec1 = self._forward(inputs, targets)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    _, pred = outputs.data.topk(1, 1, True, True)
    pred = pred.t()
    # Note:
    #     expand_as(tensor) # 将tensor扩展为参数tensor的大小
    #     eq(input, other, out=None) # 比较元素相等性
    correct = pred.eq(targets.data.view(1, -1).expand_as(pred))
    ret = []
    for k in (1,): # 如果改动了这里的100,那么上面outputs.data.topk(100, 1, True, True)中的100也要改动!!!
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        # Note:
        #     torch.mul(input, value, out=None) # 用标量值value乘以输入input的每个元素,并返回一个新的结果张量.
        ret.append(correct_k.mul_(1. / batch_size))
    prec = ret[0]
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 5 == 0:
        print('Loss {:.3f}\t'
              'Prec {:.2%}\t'
              .format(loss.item(),
                      prec.item()))
    if i == 20:
        break
'''
#top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val)
test_transformer = T.Compose([
    T.RectScale(data_height, data_width),
    T.ToTensor(),
    normalizer,
])
val_loader = DataLoader(
    Preprocessor(dataset.val, root=dataset.images_dir,
                 transform=test_transformer),
    batch_size=batch_size, num_workers=workers,
    shuffle=False, pin_memory=True)
modules = None
model.eval()
features = OrderedDict()
labels = OrderedDict()
for i, (imgs, fnames, pids, _) in enumerate(val_loader):
    # "img, fname, pid, camid".
    #features, _ = extract_features(self.model, data_loader)
    if modules is None:
        outputs = model(imgs)
        outputs = outputs.data.cpu()
    for fname, output, pid in zip(fnames, outputs, pids):
        features[fname] = output
        labels[fname] = pid
x = torch.cat([features[f].unsqueeze(0) for f, _, _ in dataset.val], 0)
y = torch.cat([features[f].unsqueeze(0) for f, _, _ in dataset.val], 0)
# Test:
#     x.size() = y.size() = torch.Size([2511, 602])
m, n = x.size(0), y.size(0)
x = x.view(m, -1)
y = y.view(n, -1)
# Note:
#     特征之间的欧几里得距离是通过(x1 - x2) ^ 2 = x1 ^ 2 + x2 ^ 2 - 2 * x1 * x2来计算的.
dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
       torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
dist.addmm_(1, -2, x, y.t())
# evaluate_all.
query_ids = [pid for _, pid, _ in dataset.val]
gallery_ids = [pid for _, pid, _ in dataset.val]
query_cams = [cam for _, _, cam in dataset.val]
gallery_cams = [cam for _, _, cam in dataset.val]
# mean_ap
dist = dist.numpy()
m, n = dist.shape
query_ids = np.asarray(query_ids)
gallery_ids = np.asarray(gallery_ids)
query_cams = np.asarray(query_cams)
gallery_cams = np.asarray(gallery_cams)
indices = np.argsort(dist, axis=1)
matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
aps = []
for i in range(m):
    # Filter out the same id and same camera
    valid = ((gallery_ids[indices[i]] != query_ids[i]) |
             (gallery_cams[indices[i]] != query_cams[i]))
    y_true = matches[i, valid]
    y_score = -dist[i][indices[i]][valid]
    if not np.any(y_true): continue
    aps.append(average_precision_score(y_true, y_score))
mAP = np.mean(aps)
