# PexelsCustom-1M: A Comprehensive Ecosystem for Open-Domain Customized Video Generation

[Paper]() | [Project Page]() | [Dataset]() | [Model Weights]()

## Overview

Recent progress in video generation has shown impressive visual synthesis capabilities. However, open-domain customized video generation remains limited by the lack of large-scale, annotated datasets capturing diverse identity-specific attributes. To address this, we introduce **PexelsCustom-1M**, the first publicly available million-scale dataset for identity-preserving video generation, containing one million curated *(identity, text, video)* triplets across **8,373 categories**.

Built on PexelsCustom-1M, we propose **CustomDiT**, a parameter-efficient Diffusion Transformer framework for customized video generation (CVG). CustomDiT conditions text-to-video generation on identity-aware reference images via bias-injected RoPE embeddings, while LoRA layers enable efficient adaptation with minimal additional parameters. Experiments demonstrate superior performance over existing methods on standard CVG benchmarks, together with improved efficiency.

Existing CVG benchmarks cover only 100 classes, limiting generalization assessment. To address this, we propose **OpenCustom**, a comprehensive evaluation suite spanning 1,000+ categories by fusing ImageNet-1K and MS-COCO. OpenCustom provides a unified protocol for identity extraction, context-aware prompting, and multi-dimensional evaluation. Extensive experiments on both prior and our new benchmark demonstrate the superiority of PexelsCustom-1M and CustomDiT.

## Installation

We use three separate conda environments for different components.

### Training and Inference (`customdit`)

```bash
conda create -n customdit python=3.10 -y
conda activate customdit
pip install -r customdit/requirements.txt
```

### Data Curation Pipeline (`data_curation`)

```bash
conda create -n data_curation python=3.10 -y
conda activate data_curation
pip install -r data_curation/requirements.txt
```

Additionally, install SAM2 and GroundingDINO from source following their respective repositories:
- [SAM 2](https://github.com/facebookresearch/segment-anything-2)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

### Evaluation (`pt2v_eval`)

```bash
conda create -n pt2v_eval python=3.10 -y
conda activate pt2v_eval
pip install -r evaluation/requirements.txt
pip install git+https://github.com/Vchitect/VBench.git
```

Also install SAM2 and GroundingDINO as described above.

## Quick Start -- Inference

1. Download the LoRA weights and place them locally (e.g., `./weights/pytorch_lora_weights.safetensors`).

2. Prepare a CSV file with columns `prompt` and `image_path`, where each row specifies a text prompt and the path to a reference image of the target identity.

3. Run inference:

```bash
cd customdit
bash inference.sh /path/to/pytorch_lora_weights.safetensors /path/to/samples.csv ./output
```

Generated videos will be saved to the specified output directory.

## Quick Start -- Training

CustomDiT training follows a two-stage procedure based on CogVideoX-5b:

**Stage 1: Without Data Augmentation (WoDA)** -- Train for 8,000 steps to learn identity-conditioned generation.

```bash
cd customdit
accelerate launch --config_file configs/accelerate_single_node.yaml train.py --config configs/train.yaml
```

**Stage 2: With Data Augmentation (WtDA)** -- Fine-tune for an additional 2,000 steps with random resizing, rotation, and shifting of reference images to mitigate the copy-paste problem.

```bash
# Update resume_path in configs/train_stage2_da.yaml to point to the Stage 1 checkpoint
accelerate launch --config_file configs/accelerate_single_node.yaml train.py --config configs/train_stage2_da.yaml
```

## Data Curation Pipeline

The data curation pipeline transforms raw Pexels videos into training-ready *(identity, text, video)* triplets through:

1. **Preprocessing**: Keyframe extraction, Florence-2 captioning, GPT-4o noun extraction, GroundingDINO detection, and SAM2 segmentation.
2. **Filtering**: Multi-stage quality filtering (aesthetic, bounding box size, overlap, SAM score).
3. **Re-captioning**: GPT-4o subject-centric caption generation with special token insertion.
4. **Post-processing**: Result aggregation, CSV merging, and train/val splitting.

For full details, see [data_curation/README.md](data_curation/README.md).

## Evaluation (OpenCustom Benchmark)

OpenCustom is a comprehensive evaluation benchmark spanning 1,000+ categories, constructed by fusing ImageNet-1K and MS-COCO. The evaluation suite measures both identity preservation and generation quality through eight metrics:

| Metric | Description |
|--------|-------------|
| CLIP-T / Regional CLIP-T | Text-image alignment (full frame / subject region) |
| CLIP-I / Regional CLIP-I | Reference-image similarity (full frame / subject region) |
| DINO-I / Regional DINO-I | Reference-image similarity via DINO (full frame / subject region) |
| Motion Smoothness | Temporal smoothness of generated videos |
| Dynamic Degree | Amount of motion in generated videos |

For full details, see [evaluation/README.md](evaluation/README.md).

## Dataset

The PexelsCustom-1M dataset will be available on HuggingFace: [Coming Soon]()

The release includes:
- **Metadata CSV**: Video IDs, subject categories, captions, and annotation references.
- **Reference images**: Cropped subject reference images.
- **Annotations**: Bounding boxes and segmentation masks (packaged as tar shards).

**Videos** are hosted on [Pexels](https://www.pexels.com) and can be downloaded using the video ID:
```
https://www.pexels.com/video/{VIDEO_ID}/
```

## Citation

```bibtex
@inproceedings{zhang2026pexelscustom,
  title={A Comprehensive Ecosystem for Open-Domain Customized Video Generation},
  author={Zhang, Jingxu and Hong, Yuqian and Kim, Daneul and Qiu, Kai and Dai, Qi and Bao, Jianmin and Yang, Yifan and Sun, Xiaoyan and Luo, Chong},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## Acknowledgments

This project builds upon several excellent open-source works:
- [CogVideoX](https://github.com/THUDM/CogVideo) -- Base video generation model
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) -- Grounding and segmentation
- [VBench](https://github.com/Vchitect/VBench) -- Video generation evaluation
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base) -- Vision-language captioning

## License

See [LICENSE](LICENSE) for details.
