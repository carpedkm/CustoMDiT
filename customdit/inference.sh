#!/bin/bash
# CustomDiT Inference
set -e
LORA_PATH=${1:-"path/to/pytorch_lora_weights.safetensors"}
CSV_PATH=${2:-"samples.csv"}
OUTPUT_DIR=${3:-"./output"}

python inference.py \
    --model_path THUDM/CogVideoX-5b \
    --lora_path $LORA_PATH \
    --csv_path $CSV_PATH \
    --output_dir $OUTPUT_DIR \
    --lora_rank 128 \
    --num_inference_steps 50 \
    --guidance_scale 6.0 \
    --num_frames 49 \
    --height 480 --width 720 \
    --fps 8 --seed 42
