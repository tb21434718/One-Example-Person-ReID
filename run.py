from __future__ import print_function, absolute_import

import os

from reid import datasets

name = 'DukeMTMC_reID'
root = os.getcwd()

dataset = datasets.create('dukemtmc_reID', os.path.join(root, 'data', name))