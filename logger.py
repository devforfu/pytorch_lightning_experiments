import os
import re
from typing import Optional, Any, Dict, Union

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
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if (step % self._log_freq) != 0:
            return
        opt_state = metrics.get('opt_state', {})
        for key, value in opt_state.items():
            opt, group, param = key.split('__')
            title = f'{param} ({opt}.{group})'
            self.vis.line(
                X=[step], Y=[value], win=key, name=param,
                opts=dict(title=title), update='append'
            )
        for index, memory in gpu_params(metrics):
            self.vis.line(
                X=[step], Y=[memory], win='gpu', name=f'cuda:{index}',
                opts=dict(title='GPU Memory'), update='append'
            )
        for phase, loss in batch_loss_metrics(metrics):
            self.vis.line(
                X=[step], Y=[loss], win='loss', name=phase,
                opts=dict(title='Batch Loss'), update='append'
            )

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
