from bengali_experiments import N_CLASSES
from bengali_experiments.densenet._base import densenet
from core import AdaptiveConcatPool2d, BaseExperiment, BaseHeadNet, MultiHead


class DenseNetThreeHeads(BaseExperiment):
    def __init__(self, params, pretrained: bool = True, metrics=None):
        super().__init__(params, metrics)
        base = densenet(
            size=params.model_size,
            weights='imagenet' if pretrained else None),
        self.net = BaseHeadNet(
            base=base,
            pool=AdaptiveConcatPool2d(1),
            head=MultiHead(
                feat_dim=base.last_linear.in_features,
                n_classes=N_CLASSES,
                drop=self.hparams.pre_head_dropout))

    def forward(self, x):
        return self.net(x)
