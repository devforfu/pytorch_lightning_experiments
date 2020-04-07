from pdb import set_trace

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


class BasicMNIST(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        from argparse import ArgumentParser
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--conv1', type=int, default=32)
        parser.add_argument('--conv2', type=int, default=64)
        parser.add_argument('--fc1', type=int, default=10)
        return parser

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.conv1 = nn.Conv2d(1, params.conv1, 3, 1)
        self.conv2 = nn.Conv2d(params.conv1, params.conv2, 3, 1)
        self.fc1 = nn.Linear(params.conv2 * 7 * 7, params.fc1)
        self.fc2 = nn.Linear(params.fc1, 10)
        self.datasets = {}

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x

    def prepare_data(self):
        transform = Compose([ToTensor()])
        mnist = MNIST(root=self.params.mnist_root, train=True,
                      download=True, transform=transform)
        train, valid = random_split(mnist, [55000, 5000])
        self.datasets['train'] = train
        self.datasets['valid'] = valid

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.params.lr)

    def train_dataloader(self):
        return DataLoader(self.datasets['train'],
                          num_workers=self.params.num_workers,
                          shuffle=True,
                          pin_memory=self.on_gpu,
                          drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.datasets['valid'],
                          num_workers=self.params.num_workers,
                          shuffle=False,
                          pin_memory=self.on_gpu,
                          drop_last=False)

    def training_step(self, batch, batch_no):
        x, y = batch
        out = self(x)
        loss = F.nll_loss(out, y)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_no):
        x, y = batch
        out = self(x)
        loss = F.nll_loss(out, y)
        logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': logs}

    def training_step_end(self, *args, **kwargs):
        set_trace()
        return
