#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = mosaic

    def __iter__(self):
        # super().__iter__()是父类的
        for batch in super().__iter__():
            yield [(self.mosaic, idx) for idx in batch]


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    # If a container object’s __iter__() method is implemented as a generator, it will automatically return an iterator object (technically, 
    # a generator object) supplying the __iter__() and __next__() methods.
    def __iter__(self):  # Return the iterator object itself. 就是一个迭代器
        start = self._rank
        # itertools.islice：创建一个迭代器，返回从 iterable 里选中的元素。如果 start 不是0，跳过 iterable 中的元素，直到到达 start 这个位置
        # 其中的参数step为self._world_size
        # https://blog.csdn.net/qq_27825451/article/details/85244237
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:  # 因为有__len__，即使while True也没事
            if self._shuffle:
                # torch.randperm：Returns a random permutation of integers from 0 to n - 1
                # 因为seed不变，所以生成的结果每次都不变
                yield from torch.randperm(self._size, generator=g)  # generator – a pseudorandom number generator for sampling
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size  # 剩下的舍弃掉
