#!/bin/bash
# CustomDiT Training
# Stage 1: Without data augmentation
# Stage 2: With data augmentation (uncomment below after Stage 1)
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage 1
accelerate launch --config_file configs/accelerate_single_node.yaml train.py --config configs/train.yaml

# Stage 2 (uncomment after Stage 1 completes, update resume_path in train_stage2_da.yaml)
# accelerate launch --config_file configs/accelerate_single_node.yaml train.py --config configs/train_stage2_da.yaml
