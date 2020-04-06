import argparse

from psutil import cpu_count

import experiments


def default_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', required=True, type=experiments.get)
    p.add_argument('--epochs', default=1, type=int)
    p.add_argument('--batch-size', default=4, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--half-precision', action='store_true')
    p.add_argument('--num-workers', default=cpu_count(), type=int)
    return p
