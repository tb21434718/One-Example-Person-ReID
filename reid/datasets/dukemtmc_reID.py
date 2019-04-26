# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:23:44 2019

@author: chen
"""
from __future__ import print_function, absolute_import

import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class DukeMTMC_ReID(Dataset):

    def __init__(self, root, split_id=0, download=True):
        super(DukeMTMC_ReID, self).__init__(root, split_id=split_id)
        
        self.index = -1
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load()

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import shutil
        from glob import glob
        
        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        identities = []
        all_pids = {}

        def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(osp.join(self.root, subdir, '*.jpg')))
            pids = set()
            for fpath in fpaths:
                self.index += 1
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                cam -= 1
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                pids.add(pid)
                if pid >= len(identities):
                    identities.append([[] for _ in range(8)])  # 8 camera views
                fname = ('{:04d}_{:01d}_{:05d}.jpg'.format(pid, cam, self.index))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        train_pids = register('bounding_box_train')
        gallery_pids = register('bounding_box_test')
        query_pids = register('query')

        # Save meta information into a json file
        meta = {'name': 'dukemtmc_reID', 'shot': 'multiple', 'num_cameras': 8,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'train': sorted(list(train_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))