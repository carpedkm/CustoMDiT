# OpenCustom Evaluation

Evaluation suite for the OpenCustom benchmark on personalized text-to-video generation.

## Environment Setup

```bash
pip install -r requirements.txt
# VBench (for motion smoothness / dynamic degree)
pip install git+https://github.com/Vchitect/VBench.git
```

You also need:
- **SAM2**: Install [SAM 2](https://github.com/facebookresearch/sam2) and download `sam2.1_hiera_large.pt` to `./checkpoints/`.
- **GroundingDINO**: Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and download `groundingdino_swint_ogc.pth` to `./gdino_checkpoints/`.

## Benchmark Data Preparation

```bash
bash prepare_benchmark.sh ./benchmark_data
```

This downloads COCO val2017 and DreamBooth. ImageNet requires manual download (see script output for instructions).

## Evaluation Workflow

Evaluation is a two-step process:

### Step 1: Segment Videos and Reference Images

Extract subject regions from generated videos and reference images using GroundingDINO + SAM2:

```bash
bash segment_videos.sh <video_dir> <image_dir> <benchmark>
# benchmark: coco | imagenet | dreambooth
```

### Step 2: Run Evaluation

Compute all metrics across the segmented data:

```bash
bash evaluate.sh <video_dir> <image_dir> <benchmark>
```

Results are saved to `./results/<benchmark>/`.

## Metrics

| Metric | Description |
|--------|-------------|
| **CLIP-T** | CLIP text-image similarity between prompt and generated frames |
| **CLIP-I** | CLIP image similarity between reference and generated frames |
| **DINO-I** | DINO image similarity between reference and generated frames |
| **Regional CLIP-I** | CLIP similarity on segmented subject regions |
| **Regional CLIP-T** | CLIP text similarity on segmented subject regions |
| **Regional DINO-I** | DINO similarity on segmented subject regions |
| **Motion Smoothness** | Temporal smoothness of generated videos |
| **Dynamic Degree** | Amount of motion/dynamics in generated videos |

## Configuration

- `NUM_GPUS`: Number of GPUs for distributed evaluation (default: 4). Set via environment variable.
- Benchmark mappings are in `benchmark/` (class, caption, and video mappings for each dataset split).
