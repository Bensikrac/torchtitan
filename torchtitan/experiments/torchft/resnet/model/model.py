# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ResNet models for TorchFT semi-sync training (CompressLoCo).

Ported from the old code-submit fork (``experiments/ft/resnet/model/``),
adapted to the new model protocol:

- ``ResNetModel`` extends ``BaseModel`` and carries its hyperparameters in a
  nested ``Config`` (the old ``ResNetModelArgs``/``FaultTolerantModelSpec``
  mechanism is gone).
- The old ``infra/parallelize.py`` FSDP/DDP logic moved into
  ``ResNetModel._apply_fsdp`` (the trainer now calls ``model.parallelize``).
- The old ``ResNetCrossEntropyLoss`` (sum-reduction cross entropy) maps to the
  shared ``torchtitan.components.loss.CrossEntropyLoss`` and is no longer
  needed here.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.models.base import BaseModel


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class ResNetModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        num_classes: int = 10
        block: Literal["ResidualBlock", "BottleNeckBlock"] = "ResidualBlock"
        layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])

        def build(self) -> "ResNetModel":
            return ResNetModel(self)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, float]:
            nparams = sum(p.numel() for p in model.parameters())

            nflops = 0.0

            in_channels = 3
            h, w = seq_len, seq_len

            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    k_h, k_w = layer.kernel_size
                    s_h, s_w = layer.stride
                    p_h, p_w = layer.padding
                    g = layer.groups
                    out_channels = layer.out_channels

                    # compute output spatial dimensions
                    h_out = math.floor((h + 2 * p_h - k_h) / s_h + 1)
                    w_out = math.floor((w + 2 * p_w - k_w) / s_w + 1)

                    # FLOPs for this conv: 2 * Cout * Hout * Wout * (Cin/group) * Kh * Kw
                    nflops += (
                        2 * out_channels * h_out * w_out * (in_channels / g) * k_h * k_w
                    )

                    # output of this layer will be input to next
                    in_channels = out_channels
                    h, w = h_out, w_out

                elif isinstance(layer, nn.Linear):
                    nflops += 2 * layer.in_features * layer.out_features

            return nparams, nflops

    def __init__(self, config: "ResNetModel.Config"):
        super().__init__()

        block = (
            ResidualBlock if config.block == "ResidualBlock" else BottleNeckBlock
        )

        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = torch.nn.ModuleDict()
        self.layers["0"] = self._make_layer(block, 64, config.layers[0], stride=1)
        self.layers["1"] = self._make_layer(block, 128, config.layers[1], stride=2)
        self.layers["2"] = self._make_layer(block, 256, config.layers[2], stride=2)
        self.layers["3"] = self._make_layer(block, 512, config.layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, config.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                if module.affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-initialize the last BN in each residual block to improve stability.
        for layer in self.layers.values():
            for block in layer:
                final_bn = getattr(block, "bn3", None)
                if final_bn is None:
                    final_bn = getattr(block, "bn2", None)

                if isinstance(final_bn, nn.BatchNorm2d) and final_bn.affine:
                    nn.init.zeros_(final_bn.weight)

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # ResNet needs no attention masks, CP sharding or SPMD annotation.
        return input_dict["input"], input_dict["labels"], {}

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
        )

        from torchtitan.distributed.fsdp import resolve_fsdp_mesh

        dp_mesh, _dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
        if training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        # For PP, by default do not reshard after forward to avoid per-microbatch
        # all-gathers, which can be expensive and non-overlapped.
        reshard_after_forward = not parallelism_context.pp_enabled

        for _layer_id, block in self.layers.items():
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        fully_shard(self, **fsdp_config)

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer in self.layers.values():
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
