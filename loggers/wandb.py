import os
from typing import Optional, List, Dict

import wandb
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_only
from torch import nn
from wandb.wandb_run import Run


class WandbLogger(LightningLoggerBase):
    def __init__(
            self,
            name: Optional[str] = None,
            save_dir: Optional[str] = None,
            offline: bool = False,
            run_id: Optional[str] = None,
            anonymous: bool = False,
            version: Optional[str] = None,
            project: Optional[str] = None,
            tags: Optional[List[str]] = None,
            experiment=None,
            entity=None
    ):
        super().__init__()
        self._name = name
        self._save_dir = save_dir
        self._anonymous = 'allow' if anonymous else None
        self._id = version or run_id
        self._tags = tags
        self._project = project
        self._experiment = experiment
        self._offline = offline
        self._entity = entity

    def __getstate__(self):
        state = self.__dict__.copy()
        # cannot be pickled
        state['_experiment'] = None
        # args needed to reload correct experiment
        state['_id'] = self.experiment.id
        return state

    @property
    def experiment(self) -> Run:
        if self._experiment is None:
            if self._offline:
                os.environ['WANDB_MODE'] = 'dryrun'
            self._experiment = wandb.init(
                name=self._name,
                dir=self._save_dir,
                project=self._project,
                anonymous=self._anonymous,
                id=self._id,
                resume=False,
                tags=self._tags,
                entity=self._entity)
        return self._experiment

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        wandb.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Dict):
        params = self._convert_params(params)
        self.experiment.config.update(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is not None:
            metrics['global_step'] = step
        self.experiment.log(metrics)

    @rank_zero_only
    def finalize(self, status: str = 'success'):
        try:
            exit_code = 0 if status == 'success' else 1
            wandb.join(exit_code)
        except TypeError:
            wandb.join()

    @property
    def name(self) -> str:
        return self.experiment.project_name()

    @property
    def version(self) -> str:
        return self.experiment.id
