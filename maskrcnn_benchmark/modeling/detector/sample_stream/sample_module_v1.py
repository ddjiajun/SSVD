import torch
import torch.nn as nn
import torch.nn.functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DeformConv

import math

class SampleBlock(nn.Module):
    def __init__(self, out_ch):
        super(SampleBlock, self).__init__()
        self.out_ch = out_ch
        
        fan_1 = float(out_ch * 2)
        std_1 = 1. / math.sqrt(fan_1)
        bound_1 = math.sqrt(3.0) * std_1
        self.off_conv1_weight = nn.Parameter(torch.randn(256, out_ch*2, 3, 3).uniform_(-bound_1, bound_1))

        fan_2 = 256.
        std_2 = 1. / math.sqrt(fan_2)
        bound_2 = math.sqrt(3.0) * std_2
        self.off_conv2_weight = nn.Parameter(torch.randn(128, 256, 3, 3).uniform_(-bound_2, bound_2))

        fan_3 = 128.
        std_3 = 1. / math.sqrt(fan_3)
        bound_3 = math.sqrt(3.0) * std_3
        self.off_conv3_weight = nn.Parameter(torch.randn(128, 128, 3, 3).uniform_(-bound_3, bound_3))

        self.offset_lateral = Conv2d(512, 256, kernel_size=(3, 3), stride=1, groups=1, dilation=1, padding=1)
        self.offset_pred = Conv2d(256, 73, kernel_size=(3, 3), stride=1, groups=1, dilation=1, padding=1)

        self.sample_conv = DeformConv(out_ch, out_ch, kernel_size=(3, 3), stride=1, groups=1, dilation=1, padding=1, deformable_groups=4, bias=False)

        # Initialization
        for modules in [self.offset_lateral, self.offset_pred, self.sample_conv]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    # print(l.weight.shape)
                    # input()
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    torch.nn.init.constant_(l.bias, 0)

        torch.nn.init.kaiming_uniform_(self.off_conv1_weight, a=1)
        torch.nn.init.kaiming_uniform_(self.off_conv2_weight, a=1)
        torch.nn.init.kaiming_uniform_(self.off_conv3_weight, a=1)
    
    def forward(self, cur_feat, sup_feat, idx):
        feat_cat = torch.cat([cur_feat, sup_feat], dim=1)

        if idx in [0, 1]:
            conv1 = F.conv2d(feat_cat, self.off_conv1_weight, bias=None, stride=1, padding=1, dilation=1, groups=1)

            conv2 = F.conv2d(conv1, self.off_conv2_weight, bias=None, stride=2, padding=1, dilation=1, groups=1)

            conv3 = F.conv2d(conv2, self.off_conv3_weight, bias=None, stride=2, padding=1, dilation=1, groups=1)

            conv2 = F.interpolate(conv2, scale_factor=2, mode='nearest')
            conv3 = F.interpolate(conv3, scale_factor=4, mode='nearest')

        elif idx in [2, 3]:
            conv1 = F.conv2d(feat_cat, self.off_conv1_weight, bias=None, stride=1, padding=1, dilation=1, groups=1)

            conv2 = F.conv2d(conv1, self.off_conv2_weight, bias=None, stride=1, padding=2, dilation=2, groups=1)

            conv3 = F.conv2d(conv2, self.off_conv3_weight, bias=None, stride=1, padding=2, dilation=2, groups=1)


        concat = torch.cat([conv1, conv2, conv3], dim=1)
        concat = F.relu(concat)
        lateral = self.offset_lateral(concat)
        lateral = F.relu(lateral)
        offset_weight = self.offset_pred(lateral)
        offset, weight = torch.split(offset_weight, [72, 1], dim=1)
        sampled_feat = self.sample_conv(sup_feat, offset)
        weight = F.relu(weight)
        return sampled_feat, weight

class SampleModule_v1(nn.Module):
    def __init__(self, cfg):
        super(SampleModule_v1, self).__init__()
        self.out_ch = cfg.MODEL.SAMPLE_STREAM.OUT_CHANNELS
        self.sample_block = SampleBlock(self.out_ch)
        
    def forward(self, x):
        if self.training:
            return self._forward_train(x)
    
    def _forward_train(self, feature_list):
        sample_features = []
        for idx, cat_feat in enumerate(feature_list):
            cur_feat, bef_feat, aft_feat = torch.chunk(cat_feat, 3, dim=0)
            bef_feat, bef_weight = self.sample_block(cur_feat, bef_feat, idx)
            aft_feat, aft_weight = self.sample_block(cur_feat, aft_feat, idx)
            weight = torch.cat([bef_weight, aft_weight], dim=1)
            weight = F.softmax(weight, dim=1)
            weight_bef, weight_aft = torch.chunk(weight, 2, dim=1)
            weight_bef = weight_bef.expand(-1, self.out_ch, -1, -1)
            weight_aft = weight_aft.expand(-1, self.out_ch, -1, -1)
            out_feat = weight_bef * bef_feat + weight_aft * aft_feat
            sample_features.append(out_feat)
        
        return sample_features

    def inference(self, cur_feat_list, sup_feat_list):
        sample_features = []
        for idx, (cur_feat, sup_feat) in enumerate(zip(cur_feat_list, sup_feat_list)):
            cur_feat = cur_feat.expand(sup_feat.shape[0], -1, -1, -1)
            sup_feat, weight = self.sample_block(cur_feat, sup_feat, idx)
            weight = F.softmax(weight, dim=0)
            weight = weight.expand(-1, self.out_ch, -1, -1)
            feat = torch.sum(weight*sup_feat, dim=0, keepdim=True)
            sample_features.append(feat)
        
        return sample_features