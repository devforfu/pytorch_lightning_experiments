from abc import ABC
from collections import OrderedDict, defaultdict

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


def to_np(tensor):
    return tensor.detach().cpu().numpy()


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


class BasicMNIST(BaseExperiment):

    def __init__(self, params, metrics=None):
        super().__init__(params, metrics)
        arch = params.arch
        self.conv1 = nn.Conv2d(1, arch.conv1, 3, 2, 1)
        self.conv2 = nn.Conv2d(arch.conv1, arch.conv2, 3, 2, 1)
        self.fc1 = nn.Linear(arch.conv2 * 7 * 7, arch.fc1)
        self.fc2 = nn.Linear(arch.fc1, 10)
        self.datasets = {}
        self.epoch_metrics = defaultdict(list)
        self.hparams = params
        self.metrics = metrics or []

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def prepare_data(self):
        transform = Compose([ToTensor()])
        mnist = MNIST(root=self.hparams.workdir, train=True,
                      download=True, transform=transform)
        train, valid = random_split(mnist, [55000, 5000])
        self.datasets['train'] = train
        self.datasets['valid'] = valid

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(self.datasets['train'],
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          pin_memory=self.on_gpu,
                          drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.datasets['valid'],
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          pin_memory=self.on_gpu,
                          drop_last=False)

    def training_step(self, batch, batch_no):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        logs = self.make_batch_logs(out, y, loss)
        self.epoch_metrics['train'].append(logs['metrics'])
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_no):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        logs = self.make_batch_logs(out, y, loss)
        self.epoch_metrics['valid'].append(logs['metrics'])
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: list):
        logs = self.make_epoch_logs()
        avg_metrics = logs.pop('average_flatten')
        avg_metrics['log'] = logs
        return avg_metrics
