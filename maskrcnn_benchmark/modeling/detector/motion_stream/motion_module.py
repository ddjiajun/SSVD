import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MotionModule(nn.Module):
    def __init__(self, cfg):
        super(MotionModule, self).__init__()

        self.out_ch = cfg.MODEL.MOTION_STREAM.OUT_CHANNELS
    
    def warp(self, x, flo, level):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        device = x.device
        B, C, H, W = x.size()
        
        if not hasattr(self, 'grid_{}'.format(level)):
            # mesh grid 
            xx = torch.arange(0, W).view(1,-1).repeat(H,1)
            yy = torch.arange(0, H).view(-1,1).repeat(1,W)
            xx = xx.view(1,1,H,W).repeat(B,1,1,1)
            yy = yy.view(1,1,H,W).repeat(B,1,1,1)
            grid = torch.cat((xx,yy),1).float().to(device)
            grid = Variable(grid)
            setattr(self, 'grid_{}'.format(level), grid) 
        else:
            grid = getattr(self, 'grid_{}'.format(level))
        
        vgrid = grid + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros')

        return output


    def inference(self, cur_feat_list, sup_feat_list, flow_list):
        
        motion_features = []
        for ind, (cur_feat, sup_feat, flow) in enumerate(zip(cur_feat_list, sup_feat_list, flow_list)):
            sup_feat = self.warp(sup_feat, flow, ind)
            feat = torch.mean(sup_feat, dim=0, keepdim=True)
            motion_features.append(feat)
        
        return motion_features
        
    
