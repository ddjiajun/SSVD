"""
SSVD framework
Modified from the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list

from .subnets import build_subnets
from .subnets import build_head_module
from .subnets import build_anchor_generator

from .backbone import build_backbone
from .backbone import build_fpn

from .sample_stream import build_sample_module
from .motion_stream import build_motion_module
from .motion_stream import build_pwc_net


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


class SSVD(nn.Module):

    def __init__(self, cfg):
        super(SSVD, self).__init__()
        num_convs_mstream = cfg.MODEL.MOTION_STREAM.NUM_CONVS
        num_convs_sstream = cfg.MODEL.SAMPLE_STREAM.NUM_CONVS
        ch_mstream = cfg.MODEL.MOTION_STREAM.OUT_CHANNELS
        ch_sstream = cfg.MODEL.SAMPLE_STREAM.OUT_CHANNELS

        # backbone
        self.backbone = build_backbone(cfg)

        # motion stream
        self.motion_fpn  = build_fpn(ch_mstream)
        self.flownet = build_pwc_net(cfg)
        self.motion_block = build_motion_module(cfg)
        self.motion_subnets = build_subnets(cfg, ch_mstream, num_convs_mstream)

        # sample stream
        self.sample_fpn = build_fpn(ch_sstream)
        self.sample_block = build_sample_module(cfg)
        self.sample_subnets = build_subnets(cfg, ch_sstream, num_convs_sstream)

        self.ssvd_head = build_head_module(cfg)
        self.anchor_generator = build_anchor_generator(cfg)

        
    def _generate_anchors(self, images, features):
        if not hasattr(self, 'base_anchors') or len(self.base_anchors) != len(features):
            self.base_anchors = self.anchor_generator(images, features)
    
    def get_fpn_feature(self, image):
        image = to_image_list(image)
        data  = image.tensors
        res_out_list = self.backbone(data)
        res_out_list = res_out_list[1:]
        mt_fpn_list = self.motion_fpn(res_out_list)
        sp_fpn_list = self.sample_fpn(res_out_list)

        out_feat_list = []
        for mt_feat, sp_feat in zip(mt_fpn_list, sp_fpn_list):
            cat_feat = torch.cat([mt_feat, sp_feat], dim=1)
            out_feat_list.append(cat_feat)
            
        return out_feat_list
    
    def aggregate_and_detect(self, cur_input_list, sup_input_list):
        cur_data = cur_input_list[0]
        cur_data_tensor = cur_data.tensors
        sup_data_tensor = sup_input_list[0]
        cur_data_repeat = cur_data_tensor.expand(sup_data_tensor.shape[0], -1, -1, -1)
        
        flow_data_cur = cur_data_repeat / 255.0
        flow_data_sup = sup_data_tensor / 255.0
        flow_list = self.flownet(flow_data_cur, flow_data_sup)

        cur_feat_list = cur_input_list[1:]
        sup_feat_list = sup_input_list[1:]
        
        mt_cur_feat_list = []
        mt_sup_feat_list = []
        sp_cur_feat_list = []
        sp_sup_feat_list = []
        for cur_feat_cat, sup_feat_cat in zip(cur_feat_list, sup_feat_list):
            mt_cur_feat, sp_cur_feat = torch.chunk(cur_feat_cat, 2, dim=1)
            mt_sup_feat, sp_sup_feat = torch.chunk(sup_feat_cat, 2, dim=1)
            mt_cur_feat_list.append(mt_cur_feat)
            mt_sup_feat_list.append(mt_sup_feat)
            sp_cur_feat_list.append(sp_cur_feat)
            sp_sup_feat_list.append(sp_sup_feat)

        motion_features = self.motion_block.inference(mt_cur_feat_list, mt_sup_feat_list, flow_list)
        sample_features = self.sample_block.inference(sp_cur_feat_list, sp_sup_feat_list)

        motion_cls_list, motion_reg_list = self.motion_subnets(motion_features)
        sample_cls_list, sample_reg_list = self.sample_subnets(sample_features)

        self._generate_anchors(cur_data, motion_features)

        result, _ = self.ssvd_head(self.base_anchors, motion_cls_list, motion_reg_list, sample_cls_list, sample_reg_list)
        # print(result[0].bbox)
        # print(result[0].get_field('labels'))
        # input()
        return result

