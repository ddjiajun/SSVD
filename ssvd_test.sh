#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NGPUS=8

cfg_file=./experiments/ssvd_test_1.yaml

python -m torch.distributed.launch  --master_port 555  --nproc_per_node=$NGPUS ./test_ssvd.py --config-file $cfg_file
