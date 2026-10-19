# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchFT Qwen 3 recipes.

Ported from the old code-submit fork (experiments/ft/qwen3/config_registry.py:
qwen3_0_6b_custom), adapted to the new recipes/config API:

- The old ``model_registry``/``FaultTolerantModelSpec`` mechanism is gone; the
  model is now built via ``torchtitan.models.qwen3.build_model_config`` and the
  generic ``FaultTolerantTrainer`` handles fragmentation via
  ``fault_tolerance.num_fragments``.
- ``FTOptimizersContainer`` was renamed to ``TorchFTOptimizersContainer``.
- ``HuggingFaceTextDataLoader`` is gone; use ``GrainDataLoader`` + ``DATASETS``.
"""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer
from torchtitan.components.validator import Validator
from torchtitan.config import CommConfig, DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.optimizer import (
    TorchFTOptim,
    TorchFTOptimizersContainer,
)
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor


def qwen3_0_6b_custom() -> FaultTolerantTrainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        pipeline_parallel_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
    )
    model_config = build_model_config("0.6B", parallelism=parallelism)
    # old config: local_batch_size=6, seq_len=4096
    num_tokens_per_microbatch_per_dp_rank = 6 * model_config.max_context_length
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        model=model_config,
        optim=TorchFTOptim.Config(
            optimizer=TorchFTOptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10000),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=model_config.max_context_length,
            steps=50000,
            dtype="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=num_tokens_per_microbatch_per_dp_rank,
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        parallelism=parallelism,
        # keep comm timeout above fault_tolerance.process_group_timeout_ms (100s)
        comm=CommConfig(train_timeout_seconds=600),
        fault_tolerance=FaultTolerance(
            enable=True,
            replica_id=0,
            group_size=1,
            semi_sync_method="diloco",
            process_group="nccl",
            process_group_timeout_ms=100000,
            sync_steps=20,
            num_fragments=1,
            # this controls how many steps until truly blocking
            fragment_sync_delay=1,
        ),
        debug=DebugConfig(
            seed=3407,
            deterministic=True,
        ),
        validator=Validator.Config(
            freq=10,
            steps=10,
        ),
    )
