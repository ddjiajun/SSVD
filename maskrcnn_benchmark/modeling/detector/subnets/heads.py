import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import  make_ssvd_postprocessor

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

class SSVDHeadModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg):
        super(SSVDHeadModule, self).__init__()
        self.cfg = cfg.clone()

        box_coder = BoxCoder(weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS)
        box_selector_test = make_ssvd_postprocessor(cfg, box_coder, is_train=False)

        self.box_selector_test = box_selector_test
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        self.nms_thresh  = cfg.MODEL.RETINANET.NMS_TH
        self.fpn_post_nms_top_n = cfg.TEST.DETECTIONS_PER_IMG


    def forward(self, anchors, motion_cls, motion_reg, sample_cls, sample_reg):
        
        return self._forward_test(anchors, motion_cls, motion_reg, sample_cls, sample_reg)

    def _forward_test(self, anchors, cls_1, reg_1, cls_2, reg_2):
        sp_boxlists = self.box_selector_test(anchors, cls_2, reg_2)
        mt_boxlists = self.box_selector_test(anchors, cls_1, reg_1)
        boxlists = cat_boxlist([mt_boxlists[0], sp_boxlists[0]])
        result = self.select_over_all_levels(boxlists)

        return result, {}

    def select_over_all_levels(self, boxlists):
        results = []
        scores = boxlists.get_field("scores")
        labels = boxlists.get_field("labels")
        boxes = boxlists.bbox
        boxlist = boxlists
        result = []
        # skip the background
        for j in range(1, self.num_classes):
            inds = (labels == j).nonzero().view(-1)

            scores_j = scores[inds]
            boxes_j = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms_thresh,
                score_field="scores"
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j,
                                        dtype=torch.int64,
                                        device=scores.device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.fpn_post_nms_top_n > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                number_of_detections - self.fpn_post_nms_top_n + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        results.append(result)

        return results