# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is copy-pasted exactly as in torch.utils.data.distributed,
# with a modification in the import to use the deprecated backend
# FIXME remove this once c10d fixes the bug it has
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import numpy as np

class DistributedSeqSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.seqlen_each_seqs = dataset.vid_length

        # self.cum_len = np.cumsum(np.array([0, np.array(self.seqlen_each_seqs)]))
        self.cum_len_all = np.cumsum(np.hstack([0, self.seqlen_each_seqs]))

        self.num_seqs = len(self.seqlen_each_seqs)

        sidx_list = [[] for _ in range(self.num_replicas)]
        cumlen_list = [[] for _ in range(self.num_replicas)]
        
        roidbs_seg_lens = np.zeros(self.num_replicas, dtype=np.int)
        for sidx in range(self.num_seqs):
            gpu_id = np.argmin(roidbs_seg_lens)
            sidx_list[gpu_id].append(sidx)
            cumlen_list[gpu_id].append(self.cum_len_all[sidx])
            roidbs_seg_lens[gpu_id] += self.seqlen_each_seqs[sidx]

        self.sidx_list = sidx_list[self.rank]
        self.num_samples = roidbs_seg_lens[self.rank]
        self.cumlen_list = cumlen_list[self.rank]
        self.seqlen_list = np.array(self.seqlen_each_seqs)[self.sidx_list]
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        # self.shuffle = True

    def __iter__(self): 
        indices = []
        
        for idx, seqlen in enumerate(self.seqlen_list):
            for fidx in range(seqlen):
                index = self.cumlen_list[idx] + fidx
                indices.append(index)

        # seqlens = np.array(self.seqlen_each_seqs)[self.sidx_list]
        # indices = [sidx for i, sidx in enumerate(self.sidx_list) for _ in range(seqlens[i]) ]
        # print(indices)
        # input()
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
