import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramid(nn.Module):
    def __init__(self, ch_out):
        super(FeaturePyramid, self).__init__()

        self.fpn_p3_1x1 = nn.Conv2d(512 , ch_out, kernel_size=1, stride=1, padding=0)
        self.fpn_p4_1x1 = nn.Conv2d(1024, ch_out, kernel_size=1, stride=1, padding=0)
        self.fpn_p5_1x1 = nn.Conv2d(2048, ch_out, kernel_size=1, stride=1, padding=0)

        self.fpn_p3 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.fpn_p4 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.fpn_p5 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.fpn_p6 = nn.Conv2d(2048, ch_out, kernel_size=3, stride=2, padding=1)

        for modules in [self.fpn_p3_1x1, self.fpn_p4_1x1, self.fpn_p5_1x1, self.fpn_p3, self.fpn_p4, self.fpn_p5, self.fpn_p6]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    torch.nn.init.constant_(l.bias, 0)                        

    def forward(self, x):
        c3, c4, c5 = x        
        
        p3_1x1 = self.fpn_p3_1x1(c3)
        p4_1x1 = self.fpn_p4_1x1(c4)
        p5_1x1 = self.fpn_p5_1x1(c5)

        p5_upsample = F.interpolate(p5_1x1, scale_factor=2, mode="nearest")
        p4_sum = p5_upsample + p4_1x1
        p4_upsample = F.interpolate(p4_sum, scale_factor=2, mode="nearest")
        p3_sum = p4_upsample + p3_1x1

        fpn_p3 = self.fpn_p3(p3_sum)
        fpn_p4 = self.fpn_p4(p4_sum)
        fpn_p5 = self.fpn_p5(p5_1x1)
        fpn_p6 = self.fpn_p6(c5)

        return fpn_p3, fpn_p4, fpn_p5, fpn_p6

def build_fpn(ch_out):
    return FeaturePyramid(ch_out)