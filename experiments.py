from argparse import ArgumentTypeError
from collections import OrderedDict
from typing import List

import pytorch_lightning as pl

from mnist_experiments.basic import BasicMNIST

EXPERIMENTS = OrderedDict()
EXPERIMENTS['mnist_basic'] = BasicMNIST


def names() -> List[str]:
    return list(EXPERIMENTS.keys())


def get(name: str) -> pl.LightningModule:
    if name not in EXPERIMENTS:
        raise ArgumentTypeError(f"'{name}' not in experiments list: {names()}")
    return EXPERIMENTS[name]
