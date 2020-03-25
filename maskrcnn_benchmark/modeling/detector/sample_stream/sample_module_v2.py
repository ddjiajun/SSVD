import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DeformConv


class SampleBlock(nn.Module):
    def __init__(self, cfg):
        super(SampleBlock, self).__init__()
        out_ch = cfg.MODEL.SAMPLE_STREAM.OUT_CHANNELS

        self.sup_conv_d1 = Conv2d(out_ch, out_ch//2, kernel_size=(3, 3), stride=1, groups=1, dilation=1, padding=1)
        self.sup_conv_d2 = Conv2d(out_ch, out_ch//4, kernel_size=(3, 3), stride=1, groups=1, dilation=2, padding=2)
        self.sup_conv_d3 = Conv2d(out_ch, out_ch//4, kernel_size=(3, 3), stride=1, groups=1, dilation=3, padding=3)

        self.offset_lateral = Conv2d(2*out_ch, out_ch, kernel_size=(3, 3), stride=1, groups=1, dilation=1, padding=1)
        self.offset_pred = Conv2d(out_ch, 73, kernel_size=(3, 3), stride=1, groups=1, dilation=1, padding=1)

        self.sample_conv = DeformConv(out_ch, out_ch, kernel_size=(3, 3), stride=1, groups=1, dilation=1, padding=1, deformable_groups=4, bias=False)

        # self.sample_conv_1x1 = Conv2d(3*256, 256, kernel_size=(1, 1), stride=1, groups=1, dilation=1, padding=0)
        
        # Initialization
        for modules in [self.sup_conv_d1, self.sup_conv_d2, self.sup_conv_d3, self.offset_lateral, self.offset_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    torch.nn.init.constant_(l.bias, 0)


    def forward(self, cur_feat, sup_feat):
        sup_feat_d1 = self.sup_conv_d1(sup_feat)
        sup_feat_d2 = self.sup_conv_d2(sup_feat)
        sup_feat_d3 = self.sup_conv_d3(sup_feat)

        feat_cat = torch.cat([cur_feat, sup_feat_d1, sup_feat_d2, sup_feat_d3], dim=1)
        feat_cat = F.relu(feat_cat)
        offset_lateral = self.offset_lateral(feat_cat)
        offset_lateral = F.relu(offset_lateral)
        offset_weight = self.offset_pred(offset_lateral)
        offset, weight = torch.split(offset_weight, [72, 1], dim=1)
        sampled_feat = self.sample_conv(sup_feat, offset)
        weight = F.relu(weight)
        return sampled_feat, weight

class SampleModule_v2(nn.Module):
    def __init__(self, cfg):
        super(SampleModule_v2, self).__init__()
        self.out_ch = cfg.MODEL.SAMPLE_STREAM.OUT_CHANNELS
        self.sample_block = SampleBlock(cfg)

    def inference(self, cur_feat_list, sup_feat_list):
        sample_features = []
        for cur_feat, sup_feat in zip(cur_feat_list, sup_feat_list):
            cur_feat = cur_feat.expand(sup_feat.shape[0], -1, -1, -1)
            sup_feat, weight = self.sample_block(cur_feat, sup_feat)
            weight = F.softmax(weight, dim=0)
            weight = weight.expand(-1, self.out_ch, -1, -1)
            feat = torch.sum(weight*sup_feat, dim=0, keepdim=True)
            sample_features.append(feat)
        
        return sample_features







