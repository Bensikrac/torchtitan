# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchFT ResNet recipes (CIFAR-10) for semi-sync training (CompressLoCo).

Ported from the old code-submit fork
(experiments/ft/resnet/config_registry.py), adapted to the new recipes/config
API. The old ``resnet18/34/50/152_cifar10`` configs referenced undefined names
(``default_ft_adamw``, ``TorchFTCheckpointManager``, ``Profiler`` were never
imported) and are ported here with those fixed:

- ``optimizer=FTOptimizersContainer.Config(lr=...)`` -> ``TorchFTOptim.Config``
  wrapping ``TorchFTOptimizersContainer`` with ``AdamW`` (renamed upstream)
- ``loss=ResNetCrossEntropyLoss.Config()`` -> shared ``CrossEntropyLoss``
  (sum reduction), the custom loss class is no longer needed
- ``local_batch_size``/``global_batch_size`` -> token-based TrainingConfig
  fields ("tokens" = images for ResNet)
- ``CheckpointManager.Config(enable=False, ...)`` -> ``checkpointer=None``
- ``Validator.Config(enable=True, ...)`` -> ``Validator.Config(...)`` (the
  ``enable`` flag is gone; pass ``validator=None`` to disable)
- ``max_norm`` gradient clipping has no equivalent in the new TrainingConfig
"""

from torchtitan.components.data.text_datasets import DATASETS  # noqa: F401
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer
from torchtitan.components.validator import Validator
from torchtitan.config import CommConfig, DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.optimizer import (
    TorchFTOptim,
    TorchFTOptimizersContainer,
)
from torchtitan.experiments.torchft.resnet import resnet_configs
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.observability.metrics import MetricsProcessor

from torchtitan.experiments.torchft.resnet import CifarDataLoader


def _resnet_config(
    *,
    flavor: str,
    lr: float,
    warmup_steps: int,
    batch_size: int,
    steps: int,
    process_group: str,
    process_group_timeout_ms: int,
    sync_steps: int,
    eval_split: bool = False,
) -> FaultTolerantTrainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        pipeline_parallel_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
    )
    # "tokens" for an image model are samples
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="",
        tokenizer=None,
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        model=resnet_configs[flavor],
        optim=TorchFTOptim.Config(
            optimizer=TorchFTOptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=lr)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=warmup_steps),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=batch_size,
            num_tokens_per_train_step=batch_size,
            steps=steps,
            dtype="float32",
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
        ),
        dataloader=CifarDataLoader.Config(
            dataset="uoft-cs/cifar10",
            batch_size=batch_size,
        ),
        checkpointer=None,
        # No activation checkpointing (the old fork used AC mode="none" for
        # ResNet); ActivationRematConfig.apply() asserts that ``layers``
        # children are Module subclasses, which ResNet's nn.Sequential stages
        # are not.
        activation_remat=None,
        parallelism=parallelism,
        # keep comm timeout above fault_tolerance.process_group_timeout_ms
        comm=CommConfig(train_timeout_seconds=600),
        fault_tolerance=FaultTolerance(
            enable=True,
            replica_id=0,
            group_size=1,
            semi_sync_method="diloco",
            process_group=process_group,
            process_group_timeout_ms=process_group_timeout_ms,
            # This needs to be specified in command line
            # sync_steps=10,
            num_fragments=1,
            # This controls how many steps until truly blocking
            fragment_sync_delay=1,
        ),
        debug=DebugConfig(
            seed=3407,
            deterministic=True,
        ),
        validator=Validator.Config(
            freq=10,
            steps=-1,
            dataloader=CifarDataLoader.Config(
                dataset="uoft-cs/cifar10",
                batch_size=batch_size,
                eval_split=eval_split,
            ),
        ),
    )


def resnet152_cifar10_custom() -> FaultTolerantTrainer.Config:
    return _resnet_config(
        flavor="152",
        lr=3e-4,
        warmup_steps=100,
        batch_size=64,
        steps=5000,
        process_group="nccl",
        process_group_timeout_ms=100000,
        sync_steps=10,
        eval_split=True,
    )


def resnet18_cifar10() -> FaultTolerantTrainer.Config:
    return _resnet_config(
        flavor="18",
        lr=0.01,
        warmup_steps=2,
        batch_size=64,
        steps=10000,
        process_group="gloo",
        process_group_timeout_ms=10000,
        sync_steps=10,
    )


def resnet34_cifar10() -> FaultTolerantTrainer.Config:
    return _resnet_config(
        flavor="34",
        lr=0.01,
        warmup_steps=2,
        batch_size=64,
        steps=10000,
        process_group="gloo",
        process_group_timeout_ms=10000,
        sync_steps=10,
    )


def resnet50_cifar10() -> FaultTolerantTrainer.Config:
    return _resnet_config(
        flavor="50",
        lr=0.01,
        warmup_steps=2,
        batch_size=64,
        steps=10000,
        process_group="gloo",
        process_group_timeout_ms=10000,
        sync_steps=10,
    )


def resnet152_cifar10() -> FaultTolerantTrainer.Config:
    return _resnet_config(
        flavor="152",
        lr=0.01,
        warmup_steps=2,
        batch_size=64,
        steps=10000,
        process_group="gloo",
        process_group_timeout_ms=10000,
        sync_steps=10,
    )
