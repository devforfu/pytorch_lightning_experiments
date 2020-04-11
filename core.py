from abc import ABC
from collections import defaultdict, OrderedDict

import numpy as np
import pytorch_lightning as pl

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
