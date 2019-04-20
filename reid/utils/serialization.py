# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:27:18 2019

@author: chen
"""
from __future__ import print_function, absolute_import

import json
import os
import torch

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def load_checkpoint(fpath):
    import re
    pattern = re.compile(r'checkpoint_(\d+)\.pt')
    fnames = os.listdir(fpath)
    for fname in fnames:
        if pattern.search(fname) == None:
            fnames.remove(fname)
    if len(fnames) == 0:
        print("No checkpoint found at {}!".format(fpath))
        return None
    else:
        fnames.sort(key = lambda x : int(x[11:-3]))
        print("Loaded checkpoint from iter step {}.".format(fnames[-1][11:-3]))
        checkpoint = torch.load(os.path.join(fpath, fnames[-1]))
        return checkpoint


def save_checkpoint(state, fpath, name):
    mkdir_if_missing(fpath)
    torch.save(state, os.path.join(fpath, name))