from .sample_module_v1 import SampleModule_v1
from .sample_module_v2 import SampleModule_v2

def build_sample_module(cfg):
    if cfg.MODEL.SAMPLE_STREAM.VERSION == '1':
        return SampleModule_v1(cfg)
    elif cfg.MODEL.SAMPLE_STREAM.VERSION == '2':
        return SamleModule_v2(cfg)
    else:
        raise ValueError("only support version 1 and 2 for sampling stream")