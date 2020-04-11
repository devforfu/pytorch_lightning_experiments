import pretrainedmodels
import torch
import torch.nn as nn


def densenet(size: int, input_layers: int = 1, weights: str = 'imagenet'):
    n = input_layers
    model_fn = pretrainedmodels.__dict__[f'densenet{size}']
    model = model_fn(num_classes=1000, pretrained=weights)
    new_conv = nn.Conv2d(n, 64, 7, 2, 3, bias=False)
    conv0 = model.features.conv0.weight
    with torch.no_grad():
        new_conv.weight[:, :] = torch.stack([torch.mean(conv0, 1)]*n, dim=1)
    model.features.conv0 = new_conv
    return model
