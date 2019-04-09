from __future__ import absolute_import

from .cuhk03 import CUHK03
from .dukemtmc_reID import DukeMTMC_ReID
from .dukemtmc_videoReID import DukeMTMC_VideoReID
from .market1501 import Market1501
from .mars import MARS


__factory = {
    'cuhk03': CUHK03,
    'dukemtmc_reID': DukeMTMC_ReID,
    'dukemtmc_videoReID': DukeMTMC_VideoReID,
    'market1501': Market1501,
    'mars': MARS,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'cuhk03', 'dukemtmc_reID',
        'dukemtmc_videoReID', 'market1501', and 'mars'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)