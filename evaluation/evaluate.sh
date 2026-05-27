#!/bin/bash
# OpenCustom Benchmark Evaluation
set -e
VIDEO_DIR=${1:?Usage: evaluate.sh <video_dir> <image_dir> <benchmark: coco|imagenet|dreambooth>}
IMAGE_DIR=${2:?}
BENCHMARK=${3:-coco}
OUTPUT_PATH=${4:-./results/${BENCHMARK}}
NUM_GPUS=${NUM_GPUS:-4}

torchrun --nproc_per_node=$NUM_GPUS --standalone evaluate.py \
    --output_path $OUTPUT_PATH \
    --video_dir $VIDEO_DIR \
    --image_dir $IMAGE_DIR \
    --json_path benchmark/video_mapping_${BENCHMARK}.json \
    --prompt_json benchmark/caption_mapping_${BENCHMARK}.json \
    --mode default \
    --n_frames 16 \
    --regional_suffix_image regional_sam
