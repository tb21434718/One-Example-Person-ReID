# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:27:18 2019

@author: chen
"""
from __future__ import print_function, absolute_import

import json
import os.path as osp

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
