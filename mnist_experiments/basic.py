from pdb import set_trace

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


class BasicMNIST(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        arch = params.arch
        self.params = params
        self.conv1 = nn.Conv2d(1, arch.conv1, 3, 2, 1)
        self.conv2 = nn.Conv2d(arch.conv1, arch.conv2, 3, 2, 1)
        self.fc1 = nn.Linear(arch.conv2 * 7 * 7, arch.fc1)
        self.fc2 = nn.Linear(arch.fc1, 10)
        self.datasets = {}

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    def prepare_data(self):
        transform = Compose([ToTensor()])
        mnist = MNIST(root=self.params.workdir, train=True,
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

    def validation_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
