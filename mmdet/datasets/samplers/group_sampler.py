# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class GroupSampler(Sampler):
    r"""Sampler that restricts data loading to the label of the dataset.

    A class-aware sampling strategy to effectively tackle the
    non-uniform class distribution. The length of the training data is
    consistent with source data. Simple improvements based on `Relay
    Backpropagation for Effective Learning of Deep Convolutional
    Neural Networks <https://arxiv.org/abs/1512.05830>`_

    The implementation logic is referred to
    https://github.com/Sense-X/TSD/blob/master/mmdet/datasets/samplers/distributed_classaware_sampler.py

    Args:
        dataset: Dataset used for sampling.
        seed (int, optional): random seed used to shuffle the sampler.
            This number should be identical across all
            processes in the distributed group. Defaults to None.
        num_sample_class (int): The number of samples taken from each
            per-label list. Defaults to 1.
    """

    def __init__(self,
                 dataset: BaseDataset,
                 samples_per_gpu: int = 2,
                 seed: Optional[int] = None) -> None:

        self.samples_per_gpu = samples_per_gpu

        assert hasattr(dataset, 'flag')
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.epoch = 0
        # Must be the same across all workers. If None, will use a
        # random seed shared among workers
        # (require synchronization among all workers)
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed

        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.world_size)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.world_size)
                ) * self.samples_per_gpu * self.world_size - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
