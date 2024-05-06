#!/bin/bash


CUDA_VISIBLE_DEVICES=0 \
singularity exec \
    --overlay /scratch/wbg231/txgnn_env/txgnn_env.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; python run.py"