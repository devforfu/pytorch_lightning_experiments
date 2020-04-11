from abc import ABC
from collections import defaultdict, OrderedDict, Callable
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils import to_np


class BaseExperiment(pl.LightningModule, ABC):

    def __init__(self, params, metrics=None):
        super().__init__()
        self.hparams = params
        self.metrics = metrics or []
        self.metrics_history = defaultdict(list)

    def make_batch_logs(self, out, y, loss):
        out, y = to_np(out), to_np(y)
        metrics = self.compute_metrics(out, y)
        metrics['loss'] = to_np(loss).item()
        logs = {'stage': 'batch',
                'metrics': metrics,
                'opt_state': self.get_optimizer_parameters()}
        return logs

    def make_epoch_logs(self):
        epoch_metrics = self.epoch_metrics.copy()
        logs = {'stage': 'epoch',
                'history': epoch_metrics,
                'current_epoch': self.trainer.current_epoch}
        avg_scores = defaultdict(dict)
        for phase, history in epoch_metrics.items():
            accum = defaultdict(list)
            for entry in history:
                for key, value in entry.items():
                    accum[key].append(value)
            for key, values in accum.items():
                avg_scores[phase][key] = np.mean(values)
        logs['average'] = dict(avg_scores)
        flatten = {}
        for phase, metrics in logs['average'].items():
            for metric, value in metrics.items():
                flatten[f'avg_{phase}_{metric}'] = value
        logs['average_flatten'] = flatten
        return logs

    def compute_metrics(self, out, y):
        return {metric.__name__: metric(out, y) for metric in self.metrics}

    def get_optimizer_parameters(self, names: tuple = ('lr', 'weight_decay')) -> dict:
        params = OrderedDict()
        for i, opt in enumerate(self.trainer.optimizers):
            for j, group in enumerate(opt.param_groups):
                for param, value in group.items():
                    if param not in names:
                        continue
                    params[f'opt_{i}__group_{j}__{param}'] = value
        return params


class AdaptiveConcatPool2d(nn.Module):
    """Applies average and maximal adaptive pooling to the tensor and
    concatenates results into a single tensor.

    The idea is taken from fastai library.
    """
    def __init__(self, size=1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(size)
        self.max = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.max(x), self.avg(x)], 1)


def create_head(feat_dim: int, n_classes: int, drop: float = 0.25):
    return nn.Sequential(
        nn.Dropout(drop),
        nn.Linear(feat_dim * 2, feat_dim),
        nn.BatchNorm1d(feat_dim),
        nn.LeakyReLU(inplace=True),
        nn.Linear(feat_dim, n_classes))


class MultiHead(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            n_classes: List,
            head_fn: Callable = create_head,
            **head_params
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            head_fn(feat_dim, n, **head_params)
            for n in n_classes])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads])


class BaseHeadNet(nn.Module):
    def __init__(self, base, pool, head, base_method='features'):
        super().__init__()
        self.base = base
        self.pool = pool
        self.head = head
        self.base_method = base_method

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x.squeeze())
        return x.squeeze()

    def features(self, x):
        return getattr(self.base, self.base_method)(x)
