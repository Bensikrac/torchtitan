# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified TorchFT Llama 3 recipes."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer
from torchtitan.config import CommConfig, DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.components.validator import Validator
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.llama3 import build_model_config
from torchtitan.experiments.torchft.optimizer import (
    TorchFTOptim,
    TorchFTOptimizersContainer,
)
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler

from torchtitan_recipes.data import c4_document_capacity


def _llama3_torchft_config(*, replica_id: int) -> FaultTolerantTrainer.Config:
    parallelism = ParallelismConfig(data_parallel_shard_degree=4)
    model_config = build_model_config("debugmodel", parallelism=parallelism)
    num_tokens_per_microbatch_per_dp_rank = 8 * model_config.max_context_length
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        optim=TorchFTOptim.Config(
            optimizer=TorchFTOptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=model_config.max_context_length,
            steps=100,
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        parallelism=parallelism,
        comm=CommConfig(train_timeout_seconds=15),
        fault_tolerance=FaultTolerance(
            enable=True,
            replica_id=replica_id,
            group_size=2,
            semi_sync_method="diloco",
            process_group="nccl",
            process_group_timeout_ms=10000,
            sync_steps=10,
            num_fragments=2,
        ),
        validator=None,
    )


def llama3_torchft_replica_0() -> FaultTolerantTrainer.Config:
    return _llama3_torchft_config(replica_id=0)


def llama3_torchft_replica_1() -> FaultTolerantTrainer.Config:
    return _llama3_torchft_config(replica_id=1)


def llama3_ft_8b_custom() -> FaultTolerantTrainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        pipeline_parallel_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
    )
    model_config = build_model_config("8B", parallelism=parallelism)
    # old config: local_batch_size=1, seq_len=4096
    num_tokens_per_microbatch_per_dp_rank = 1 * model_config.max_context_length
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        model=model_config,
        optim=TorchFTOptim.Config(
            optimizer=TorchFTOptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=6000),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            # old config: global_batch_size=48
            num_tokens_per_train_step=48 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=30000,
            dtype="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
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
