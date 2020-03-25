from .subnets import SSVDSubnets
from .heads import SSVDHeadModule
from .anchor_generator import make_anchor_generator_retinanet

def build_subnets(cfg, ch_out, num_convs):
    return SSVDSubnets(cfg, ch_out, num_convs)

def build_head_module(cfg):
    return SSVDHeadModule(cfg)

def build_anchor_generator(cfg):
    return make_anchor_generator_retinanet(cfg)
