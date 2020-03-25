from .motion_module import MotionModule
from .pwc_net import PWC_Net


def build_motion_module(cfg):
    return MotionModule(cfg)

def build_pwc_net(cfg):
    return PWC_Net(cfg)
