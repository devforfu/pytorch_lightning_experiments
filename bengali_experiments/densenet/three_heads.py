import albumentations as A

from bengali_experiments import N_CLASSES, IMG_SIZE
from bengali_experiments.densenet._base import densenet
from core import AdaptiveConcatPool2d, BaseExperiment, BaseHeadNet, MultiHead


TRANSFORM_TRAIN = A.Compose([
    A.Resize(width=IMG_SIZE[1], height=IMG_SIZE[0]),
    A.OneOf(
        [
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
        ],
        p=0.5
    ),
    A.OneOf(
        [
            A.GridDistortion(),
            A.ElasticTransform(),
            A.OpticalDistortion(),
            A.ShiftScaleRotate()
        ],
        p=0.25
    ),
    A.CoarseDropout(),
    A.Normalize(mean=0.5, std=0.5)
])

TRANSFORM_VALID = A.Compose([
    A.Resize(width=IMG_SIZE[1], height=IMG_SIZE[0]),
    A.Normalize(mean=0.5, std=0.5)
])


class BengaliDenseNetThreeHeads(BaseExperiment):
    def __init__(self, params, pretrained: bool = True, metrics=None):
        super().__init__(params, metrics)
        base = densenet(
            size=self.hparams.arch.model_size,
            weights='imagenet' if pretrained else None)
        self.net = BaseHeadNet(
            base=base,
            pool=AdaptiveConcatPool2d(1),
            head=MultiHead(
                feat_dim=base.last_linear.in_features,
                n_classes=N_CLASSES,
                drop=self.hparams.arch.pre_head_dropout))

    def forward(self, x):
        return self.net(x)
