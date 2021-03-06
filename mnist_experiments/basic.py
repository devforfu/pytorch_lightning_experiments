from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from core import BaseExperiment, recall, accuracy


class BasicMNIST(BaseExperiment):

    def __init__(self, params, metrics=None):
        super().__init__(params, metrics=(
            [recall, accuracy] if metrics is None else
            [] if metrics is False else
            metrics))
        arch = params.arch
        self.conv1 = nn.Conv2d(1, arch.conv1, 3, 2, 1)
        self.conv2 = nn.Conv2d(arch.conv1, arch.conv2, 3, 2, 1)
        self.fc1 = nn.Linear(arch.conv2 * 7 * 7, arch.fc1)
        self.fc2 = nn.Linear(arch.fc1, 10)
        self.datasets = {}
        self.epoch_metrics = defaultdict(list)
        self.hparams = params

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
        self.epoch_metrics = defaultdict(list)
        return avg_metrics
