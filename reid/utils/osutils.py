# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:27:07 2019

@author: chen
"""
from __future__ import absolute_import

import os


def mkdir_if_missing(dir_path):
    if os.path.isdir(dir_path):
        return
    else:
        os.makedirs(dir_path)