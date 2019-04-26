# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:42:08 2019

@author: chen
"""
from __future__ import print_function

import os.path as osp

from ..serialization import read_json


def _pluck(identities, indices):
    ret = []
    for pid in indices:
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                _, _, index = map(int, name.split('_'))
                ret.append((fname, pid, camid, index))
    return ret


class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.num_cameras = 0
        self.num_train_ids = 0
        self.train, self.query, self.gallery = [], [], []

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, verbose=True):
        # Read the splits.json
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        split = splits[self.split_id]
        #
        meta = read_json(osp.join(self.root, 'meta.json'))
        identities = meta['identities']
        self.num_cameras = meta['num_cameras']
        self.num_train_ids = len(split['train'])
        self.train = _pluck(identities, split['train'])
        self.query = _pluck(identities, split['query'])
        self.gallery = _pluck(identities, split['gallery'])
        #
        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("Subset   |   ID  |   Images")
            print("---------------------------")
            print("Train    | {:5d} | {:8d}"
                  .format(len(split['train']), len(self.train)))
            print("Query    | {:5d} | {:8d}"
                  .format(len(split['query']), len(self.query)))
            print("Gallery  | {:5d} | {:8d}"
                  .format(len(split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))