# Modified from maskrcnn-benchmark
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ssvd import SSVD

_DETECTION_META_ARCHITECTURES = {
                                 "SSVD": SSVD,
                                 }


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
