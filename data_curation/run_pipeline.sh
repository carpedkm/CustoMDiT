#!/usr/bin/env bash
# Master script for PexelsCustom-1M data curation pipeline.
# Runs all 5 steps sequentially.
#
# Required environment variables:
#   DATA_ROOT       - Path to raw video directory
#   METADATA_CSV    - Path to input metadata CSV
#   OUTPUT_DIR      - Pipeline output root directory
#   OPENAI_API_KEY  - OpenAI API key (or set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY)
#
# Optional:
#   NUM_GPUS        - Number of GPUs for distributed steps (default: 1)
#   N_CHUNKS        - Number of chunks for parallel processing (default: 1)
#   CHUNK_IDX       - Chunk index to process (default: 0)
#   MODEL_NAME      - OpenAI model name (default: gpt4o)

set -euo pipefail

# --- Validate required env vars ---
if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${AZURE_OPENAI_ENDPOINT:-}" ]; then
    echo "ERROR: Set OPENAI_API_KEY or (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY)"
    exit 1
fi

: "${DATA_ROOT:?Set DATA_ROOT to raw video directory}"
: "${METADATA_CSV:?Set METADATA_CSV to input metadata CSV path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to pipeline output root}"

NUM_GPUS="${NUM_GPUS:-1}"
N_CHUNKS="${N_CHUNKS:-1}"
CHUNK_IDX="${CHUNK_IDX:-0}"
MODEL_NAME="${MODEL_NAME:-gpt4o}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREPROCESS_DIR="${OUTPUT_DIR}/preprocessing"
FILTER_DIR="${OUTPUT_DIR}/filtering"
RECAP_DIR="${OUTPUT_DIR}/recaptioning"
FINAL_DIR="${OUTPUT_DIR}/final"

echo "============================================"
echo "PexelsCustom-1M Data Curation Pipeline"
echo "============================================"
echo "DATA_ROOT:    ${DATA_ROOT}"
echo "METADATA_CSV: ${METADATA_CSV}"
echo "OUTPUT_DIR:   ${OUTPUT_DIR}"
echo "NUM_GPUS:     ${NUM_GPUS}"
echo "N_CHUNKS:     ${N_CHUNKS}"
echo "CHUNK_IDX:    ${CHUNK_IDX}"
echo "============================================"

# --- Step 1: Preprocessing (Florence2 + GroundingDINO + GPT-4o) ---
echo ""
echo "[Step 1/5] Preprocessing..."
torchrun --nproc_per_node="${NUM_GPUS}" "${SCRIPT_DIR}/1_preprocessing.py" \
    --p_meta "${METADATA_CSV}" \
    --dataroot "${DATA_ROOT}" \
    --datasave "${PREPROCESS_DIR}" \
    --n_chunks "${N_CHUNKS}" \
    --chunk_idx "${CHUNK_IDX}" \
    --model_name "${MODEL_NAME}"

# --- Step 2: Segmentation (SAM2) ---
# Note: Step 2 requires a JSON metadata file mapping image paths to captions.
# Skip if not applicable to your workflow (preprocessing already includes SAM2).
# echo ""
# echo "[Step 2/5] Segmentation..."
# torchrun --nproc_per_node="${NUM_GPUS}" "${SCRIPT_DIR}/2_segmentation.py" \
#     --p_meta "${PREPROCESS_DIR}/segmentation_meta.json" \
#     --dataroot "${DATA_ROOT}" \
#     --datasave "${PREPROCESS_DIR}/segmentation"

# --- Step 3: Filtering ---
echo ""
echo "[Step 3/5] Filtering..."
python "${SCRIPT_DIR}/3_filtering.py" \
    --p_meta "${METADATA_CSV}" \
    --dataroot "${PREPROCESS_DIR}" \
    --datasave "${FILTER_DIR}" \
    --n_chunks "${N_CHUNKS}" \
    --chunk_idx "${CHUNK_IDX}" \
    --model_name "${MODEL_NAME}"

# --- Step 4: Recaptioning ---
echo ""
echo "[Step 4/5] Recaptioning..."
python "${SCRIPT_DIR}/4_recaptioning.py" \
    --p_meta "${METADATA_CSV}" \
    --dataroot "${FILTER_DIR}" \
    --datasave "${RECAP_DIR}" \
    --image_root "${PREPROCESS_DIR}" \
    --n_chunks "${N_CHUNKS}" \
    --chunk_idx "${CHUNK_IDX}" \
    --model_name "${MODEL_NAME}"

# --- Step 5: Post-processing ---
echo ""
echo "[Step 5/5] Post-processing..."
python "${SCRIPT_DIR}/5_postprocess.py" \
    --input_dir "${OUTPUT_DIR}" \
    --output_dir "${FINAL_DIR}" \
    --metadata_csv "${METADATA_CSV}" \
    --val_size 1000 \
    --recap_root_dir "${RECAP_DIR}"

echo ""
echo "============================================"
echo "Pipeline complete! Final outputs in: ${FINAL_DIR}"
echo "============================================"
