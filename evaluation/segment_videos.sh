#!/bin/bash
# Segment videos and reference images for evaluation
set -e
VIDEO_DIR=${1:?Usage: segment_videos.sh <video_dir> <image_dir> <benchmark: coco|imagenet|dreambooth>}
IMAGE_DIR=${2:?}
BENCHMARK=${3:-coco}
NUM_GPUS=${NUM_GPUS:-4}

echo "Segmenting generated videos..."
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS segment_videos.py \
    --json_path benchmark/class_mapping_${BENCHMARK}.json \
    --dataroot $VIDEO_DIR --use_sam

echo "Segmenting reference images..."
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS segment_videos.py \
    --json_path benchmark/class_mapping_${BENCHMARK}.json \
    --dataroot $IMAGE_DIR --use_sam --process_image
