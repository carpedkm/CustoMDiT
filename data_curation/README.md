# PexelsCustom-1M Data Curation Pipeline

End-to-end pipeline for processing raw Pexels videos into training-ready data with subject-centric annotations.

## Prerequisites

### Model Checkpoints

| Model | Path | Notes |
|-------|------|-------|
| SAM2 | `./checkpoints/sam2.1_hiera_large.pt` | [SAM2 repo](https://github.com/facebookresearch/segment-anything-2) |
| GroundingDINO | `./gdino_checkpoints/groundingdino_swint_ogc.pth` | [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO) |
| Florence-2 | Auto-downloaded from HuggingFace | `microsoft/Florence-2-base` |

### Python Environment

```bash
pip install -r requirements.txt
```

Also install SAM2 and GroundingDINO following their respective repos.

## Environment Setup

```bash
# Required
export DATA_ROOT=/path/to/raw/videos
export METADATA_CSV=/path/to/metadata.csv
export OUTPUT_DIR=/path/to/pipeline/output
export OPENAI_API_KEY=sk-...

# Optional
export NUM_GPUS=8
export N_CHUNKS=16        # for parallel job submission
export CHUNK_IDX=0        # which chunk to process
export MODEL_NAME=gpt4o   # OpenAI model name

# For Azure OpenAI (instead of OPENAI_API_KEY):
# export AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
# export AZURE_OPENAI_KEY=your-key
```

## Running

```bash
# Full pipeline
bash run_pipeline.sh

# Or run steps individually:
torchrun --nproc_per_node=8 1_preprocessing.py --p_meta meta.csv --dataroot /videos --datasave /output/preprocessing
python 3_filtering.py --p_meta meta.csv --dataroot /output/preprocessing --datasave /output/filtering
python 4_recaptioning.py --p_meta meta.csv --dataroot /output/filtering --datasave /output/recaptioning --image_root /output/preprocessing
python 5_postprocess.py --input_dir /output --output_dir /output/final --metadata_csv meta.csv
```

## Pipeline Steps

1. **Preprocessing** (`1_preprocessing.py`): Extract keyframes, generate Florence-2 captions, extract object nouns via GPT-4o, detect with GroundingDINO, segment with SAM2
2. **Segmentation** (`2_segmentation.py`): Standalone SAM2 segmentation for additional mask generation
3. **Filtering** (`3_filtering.py`): Multi-stage annotation filtering (human body parts, redundant classes, background objects, size, SAM score, IoU dedup)
4. **Recaptioning** (`4_recaptioning.py`): GPT-4o subject-centric caption generation with special token insertion
5. **Post-processing** (`5_postprocess.py`): Gather results, merge CSVs, train/val split

## Input Format

Metadata CSV with columns: `videoid`, `name` (caption), and other optional metadata.

## Output Format

- Per-video JSON files with annotations (bounding boxes, RLE masks, captions)
- Final `train.csv` and `val.csv` for training

## API Cost Notes

Steps 1, 3, and 4 call GPT-4o. Step 4 also sends images (higher token cost). Budget approximately:
- Step 1: ~1 text call per video
- Step 3: ~1 text call per video
- Step 4: ~1-3 vision calls per video (depends on number of objects)

For 1M videos, expect significant API costs. Use `--n_chunks` to parallelize across machines.
