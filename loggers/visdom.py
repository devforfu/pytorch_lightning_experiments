import os
import re
from collections import defaultdict
from typing import Optional, Any, Dict, Union

import torch
from pytorch_lightning.loggers import LightningLoggerBase, rank_zero_only

from visdom import Visdom


class VisdomLogger(LightningLoggerBase):

    def __init__(
            self,
            env: str = 'main',
            version: int = 1,
            username: Optional[str] = None,
            password: Optional[str] = None,
            server: Optional[str] = '0.0.0.0',
            port: Optional[int] = 9090,
            log_freq: int = 1,
            experiment=None
    ):
        super().__init__()
        self._env = env
        self._version = version
        self._username = username or os.getenv('VISDOM_USERNAME')
        self._password = password or os.getenv('VISDOM_PASSWORD')
        self._server = server
        self._port = port
        self._log_freq = log_freq
        self._experiment = experiment or self._init()

    def _init(self):
        experiment = Visdom(
            server=self._server,
            port=self._port,
            username=self._username,
            password=self._password,
            env=self._env)
        return experiment

    @property
    def vis(self):
        return self.experiment

    @property
    def experiment(self) -> Any:
        return self._experiment

    @rank_zero_only
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        if (step % self._log_freq) != 0:
            return

        if metrics['stage'] == 'batch':
            gpu_keys = [key for key in metrics if key.startswith('gpu_')]
            for key in gpu_keys:
                self.vis.line(
                    X=[step], Y=[metrics[key]], win='gpu', name=key,
                    opts=dict(title='GPU Memory'), update='append')
            opt_state = metrics.get('opt_state', {})
            for key, value in opt_state.items():
                opt, group, param = key.split('__')
                title = f'{param} ({opt}.{group})'
                self.vis.line(
                    X=[step], Y=[value], win=key, name=param,
                    opts=dict(title=title), update='append')
            performance = metrics.get('metrics', {})
            for metric_name, value in performance.items():
                self.vis.line(
                    X=[step], Y=[value], win=metric_name, name=metric_name,
                    opts=dict(title=f'{metric_name} (train batch)'),
                    update='append')

        elif metrics['stage'] == 'epoch':
            avg_scores = defaultdict(dict)
            for phase, history in metrics['history'].items():
                collected = defaultdict(list)
                for entry in history:
                    for key, value in entry.items():
                        collected[key].append(value)
                for key, values in collected.items():
                    avg_metric = torch.mean(torch.stack(values))
                    avg_scores[key][phase] = avg_metric
            phases = list(metrics['history'])
            for key in avg_scores:
                for phase in phases:
                    avg = avg_scores[key][phase]
                    self.vis.line(
                        X=[metrics['current_epoch']], Y=[avg],
                        win=f'avg_{key}', name=phase,
                        opts=dict(title=f'{key} (avg/epoch)'),
                        update='append')

        # opt_state = metrics.get('opt_state', {})
        # for key, value in opt_state.items():
        #     opt, group, param = key.split('__')
        #     title = f'{param} ({opt}.{group})'
        #     self.vis.line(
        #         X=[step], Y=[value], win=key, name=param,
        #         opts=dict(title=title), update='append'
        #     )
        # for index, memory in gpu_params(metrics):
        #     self.vis.line(
        #         X=[step], Y=[memory], win='gpu', name=f'cuda:{index}',
        #         opts=dict(title='GPU Memory'), update='append'
        #     )


    @rank_zero_only
    def log_hyperparams(self, params: Dict):
        pass

    @rank_zero_only
    def finalize(self, status: str) -> None:
        pass

    @property
    def name(self) -> str:
        return self._env

    @property
    def version(self) -> Union[int, str]:
        return self._version


def gpu_params(metrics: dict) -> list:
    return [(int(k.split('_')[-1]), v)
            for k, v in metrics.items()
            if k.startswith('gpu')]


def batch_loss_metrics(metrics: dict) -> list:
    losses = []
    for k, v in metrics.items():
        m = re.match(r'^(trn|val)_loss$', k)
        if m is not None:
            phase = m.groups()[0]
            losses.append((phase, v))
    return losses

