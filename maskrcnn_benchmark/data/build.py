# Modified from maskrcnn-benchmark
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from . import datasets as D
from . import samplers

from .collate_batch import SeqBatchCollator
from .transforms import build_test_transforms_cv2

def build_dataset(dataset_list, transforms, augmentations, dataset_catalog, data_dir, cfg, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name, data_dir)
        factory = getattr(D, data["factory"])
        args = data["args"]
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets


def make_data_sampler(dataset, shuffle, distributed, is_train=True):
    if distributed:
        return samplers.DistributedSeqSampler(dataset, shuffle=shuffle)
    else:
        return samplers.DistributedSeqSampler(dataset, num_replicas=1, rank=0, shuffle=shuffle)
            

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=False, is_distributed=False, start_iter=0):
    assert is_train == False, 'only support testing now'

    num_gpus = get_world_size()
    
    images_per_batch = cfg.TEST.IMS_PER_BATCH
    assert (
        images_per_batch % num_gpus == 0
    ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(images_per_batch, num_gpus)
    images_per_gpu = images_per_batch // num_gpus
    assert images_per_gpu == 1, 'only support test with one image per gpu'
    shuffle = False if not is_distributed else True
    num_iters = None
    start_iter = 0

    aspect_grouping = []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TEST

    # build transforms and data augmentations
    transforms = build_test_transforms_cv2(cfg)
    augmentations = None
    
    # build datasets
    datasets = build_dataset(dataset_list, transforms, augmentations, DatasetCatalog, cfg.DATA_DIR, cfg, is_train)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed, is_train)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = SeqBatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)

        num_workers = cfg.DATALOADER.NUM_WORKERS

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    
    return data_loaders
