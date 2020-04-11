from collections import OrderedDict
from typing import Tuple, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.functional import cross_entropy

import logger
from bengali_experiments import TRN_NUMPY, TRN_IMAGE_IDS, LABELS, N_CLASSES
from utils import adjacent_pairs

log = logger.get_logger()


def load_dataset(debug: bool = False, pct: float = 0.1) -> Tuple:
    log.info('Loading the dataset')
    data = np.load(TRN_NUMPY)
    ids = np.load(TRN_IMAGE_IDS)
    labels = np.load(LABELS)
    n, m, k = [len(arr) for arr in (data, ids, labels)]
    assert n == m == k, f'Should be equal: {n} == {m} == {k}'

    if debug:
        log.warning('Debugging mode: the dataset size is reduced!')
        sz = int(n * pct)
        data, ids = data[:sz], ids[:sz]
        labels = OrderedDict([
            (k, v)
            for k, v in labels.items()
            if k in ids
        ])

    return data, ids, labels


def no_aug(x): return {'image': x}


class BengaliDataset(Dataset):
    def __init__(
            self,
            data, ids, labels,
            subset: list = None,
            transform: Callable = no_aug
    ):
        self.data = data
        self.ids = ids
        self.labels = labels
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return (
            self.data.shape[0] if self.subset else
            len(self.subset))

    def __getitem__(self, index):
        self._check_index(index)
        index = self._adjust_index(index)
        image, image_id = self.data[index], self.ids[index]
        image = self.transform(image)['image']
        target = np.asarray(list(self.labels[image_id][:3]), dtype=np.float32)
        return {'image_id': image_id, 'image': image, 'target': target}

    def _check_index(self, index):
        n = len(self)
        assert 0 <= index < n, f'Index should be in the range: [0; {n})'

    def _adjust_index(self, index):
        if self.subset is None:
            return index
        return self.subset[index]


class ThreeLabelsCE(nn.Module):
    def __init__(
            self,
            n_classes: tuple = N_CLASSES,
            weights: tuple = (.5, .25, .25),
            device: torch.device = 'cpu'
    ):
        super().__init__()
        assert len(n_classes) == len(weights), 'Weights list does not match classes!'
        self.n_classes = n_classes
        self.weights = list(weights)
        self.device = device
        self.ranges = list(adjacent_pairs([0] + list(np.cumsum(n_classes))))
        self.i = 0

    def forward(self, x, targets):
        losses = []
        for i, (lo, hi) in enumerate(self.ranges):
            ce = cross_entropy(x[:, lo:hi], targets[:, i].long())
            losses.append(ce)
        loss = torch.sum(torch.stack(losses).mul(self._weights))
        return loss

    @property
    def _weights(self):
        return torch.FloatTensor(self.weights).to(self.device)
