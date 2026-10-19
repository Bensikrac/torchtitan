# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.torchft.resnet.datasets.cifar10 import CifarDataLoader
from torchtitan.experiments.torchft.resnet.model.model import ResNetModel

__all__ = [
    "CifarDataLoader",
    "ResNetModel",
]


resnet_configs = {
    "18": ResNetModel.Config(
        num_classes=10, block="ResidualBlock", layers=[2, 2, 2, 2]
    ),
    "34": ResNetModel.Config(
        num_classes=10, block="ResidualBlock", layers=[3, 4, 6, 3]
    ),
    "50": ResNetModel.Config(
        num_classes=10, block="BottleNeckBlock", layers=[3, 4, 6, 3]
    ),
    "152": ResNetModel.Config(
        num_classes=10, block="BottleNeckBlock", layers=[3, 8, 36, 3]
    ),
}
