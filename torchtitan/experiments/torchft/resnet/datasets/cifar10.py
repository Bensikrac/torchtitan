# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR-10 image dataloader for TorchFT semi-sync training (CompressLoCo).

Ported from the old code-submit fork (``experiments/ft/resnet/datasets/cifar10.py``).
The old loader extended ``ParallelAwareDataloader`` (removed upstream); it is
now a ``BaseDataLoader`` yielding ``TokenizedTrainingMicrobatch`` objects. The
``positions``/``padding_mask`` fields required by the token-based microbatch
are unused by ResNet and filled with dummies.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import DataLoader, IterableDataset

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TokenizedTrainingMicrobatch

logger = logging.getLogger(__name__)


class HuggingFaceImageDataset(IterableDataset, Stateful):
    """A simple HuggingFace image dataset for CIFAR-10.

    This dataset uses the HuggingFace datasets library to load the CIFAR-10
    dataset. DP sharding is handled by ``split_dataset_by_node``.
    """

    def __init__(
        self,
        dataset_name: str,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        eval_split: bool = False,
    ):
        self.dataset_name = dataset_name
        if eval_split:
            ds = load_dataset(self.dataset_name, split="test")
        else:
            ds = load_dataset(self.dataset_name, split="train")
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0

    def __iter__(self):
        while True:
            for item in self._data:
                image = item["img"]
                label = item["label"]

                # Convert PIL Image to numpy array → tensor
                np_img = np.array(image)
                tensor_img = torch.from_numpy(np_img).permute(
                    2, 0, 1
                )  # (H, W, C) → (C, H, W)

                # Optionally normalize
                tensor_img = tensor_img.float() / 255.0

                yield tensor_img, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def state_dict(self) -> dict[str, object]:
        return {"sample_idx": self._sample_idx}

    def load_state_dict(self, state_dict) -> None:
        self._sample_idx = state_dict.get("sample_idx", 0)


class CifarDataLoader(BaseDataLoader):
    """Configurable image dataloader for CIFAR-10 using HuggingFace datasets."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str = "uoft-cs/cifar10"
        infinite: bool = True
        eval_split: bool = False
        batch_size: int = 64

    def __init__(
        self,
        config: "CifarDataLoader.Config",
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer=None,
        max_context_length: int = 0,
        num_tokens_per_microbatch: int = 0,
        **kwargs,
    ):
        del tokenizer, max_context_length, num_tokens_per_microbatch, kwargs
        self.config = config
        dataset = HuggingFaceImageDataset(
            dataset_name=config.dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            eval_split=config.eval_split,
        )
        self._loader = DataLoader(dataset, batch_size=config.batch_size)

    def __iter__(self):
        for images, labels in self._loader:
            labels = torch.as_tensor(labels, dtype=torch.long)
            # "tokens" for an image model are samples; positions/padding_mask
            # are unused by ResNet and filled with dummies to satisfy the
            # token-based microbatch contract.
            num_positions = int(images.numel() // images.shape[0])
            yield TokenizedTrainingMicrobatch(
                input=images,
                labels=labels,
                positions=torch.arange(num_positions),
                padding_mask=torch.ones(images.shape[0], num_positions),
                loss_token_counts=torch.tensor(float(labels.shape[0])),
                routing_token_counts=torch.tensor(0.0),
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "dataset": self._loader.dataset.state_dict()
            if hasattr(self._loader.dataset, "state_dict")
            else {}
        }

    def load_state_dict(self, state_dict) -> None:
        dataset_state = state_dict.get("dataset", {})
        if hasattr(self._loader.dataset, "load_state_dict"):
            self._loader.dataset.load_state_dict(dataset_state)
