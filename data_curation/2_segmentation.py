"""Step 2: SAM2 segmentation mask generation.

This script takes preprocessing JSON outputs (with object detections) and generates
precise segmentation masks using SAM2 + GroundingDINO. For each detected object,
it produces masked images and RLE-encoded segmentation results.

Usage:
    torchrun --nproc_per_node=NUM_GPUS 2_segmentation.py \
        --p_meta input.json --dataroot /path/to/images --datasave /path/to/output
"""

import os
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP
import json
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import pandas as pd
import decord
import tqdm
import time
import grounding_dino.groundingdino.datasets.transforms as T
import traceback
import torch.distributed as dist
import random
from datetime import timedelta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_meta', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--datasave', type=str, required=True)
    parser.add_argument('--n_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--is_video', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--sam2_checkpoint', type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--sam2_model_config', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--gdino_config', type=str, default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--gdino_checkpoint', type=str, default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    args = parser.parse_args()
    return args

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def load_image(image_input):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = image_input
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

def distribute_list_to_rank(data_list):
    data_list = data_list[get_rank()::get_world_size()]
    return data_list

if __name__ == '__main__':
    args = parse_args()
    p_meta = args.p_meta
    dataroot = args.dataroot
    datasave = args.datasave
    verbose = args.verbose
    os.makedirs(datasave, exist_ok=True)
    os.makedirs(os.path.join(datasave, 'error'), exist_ok=True)
    os.makedirs(os.path.join(datasave,'mask'), exist_ok=True)
    os.makedirs(os.path.join(datasave,'mask_full'), exist_ok=True)
    os.makedirs(os.path.join(datasave, 'json'), exist_ok=True)


    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup DDP:
    global_seed=42
    torch.manual_seed(global_seed)

    dist.init_process_group("nccl", timeout=timedelta(seconds=7200000))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CONFIG = args.sam2_model_config
    GROUNDING_DINO_CONFIG = args.gdino_config
    GROUNDING_DINO_CHECKPOINT = args.gdino_checkpoint
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25


    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )

    with open(p_meta, 'r') as f:
        data = json.load(f)

    image_paths = list(data.keys())
    image_paths = distribute_list_to_rank(image_paths)

    error_files = []
    for image_path in image_paths:
        image_fp = os.path.join(dataroot, image_path)
        caption = data[image_path]

        mask_image_save_path = os.path.join(datasave, 'mask', image_path)
        masked_image_full_save_path = os.path.join(datasave, 'mask_full', image_path)
        result_path = os.path.join(datasave, 'json', image_path.rsplit('.', 1)[0] + '.json')

        image = Image.open(image_fp).convert("RGB")
        image_source, image = load_image(image)
        boxes, confidences, labels = predict(
                        model=grounding_model,
                        image=image,
                        caption=caption,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD,
                    )
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        if len(input_boxes) == 0:
            error_files.append(image_path)
            image = Image.open(image_fp).convert("RGB")
            image.save(mask_image_save_path)
            image.save(masked_image_full_save_path)
            mask = np.zeros((h, w), dtype=np.uint8)
            rle = single_mask_to_rle(mask)
            result = {
                'bbox': [0, 0, w, h],
                'segmentation': rle
            }
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)
            continue
        elif len(input_boxes) > 1:
            #Top 1 result.
            top_idx = np.argmax(confidences)
            input_boxes, confidences = input_boxes[top_idx:top_idx+1], confidences[top_idx:top_idx+1]

        sam2_predictor.set_image(image_source)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        binary_mask = masks[0]
        img_annotate = image_source
        masked_image = np.where(binary_mask[:, :, None], img_annotate, np.array([127, 127, 127], dtype=img_annotate.dtype))


        Image.fromarray(masked_image.astype(np.uint8)).save(mask_image_save_path)
        masked_image_full = np.where(binary_mask[:, :, None], img_annotate, np.array([0, 0, 0], dtype=img_annotate.dtype))
        masked_image_full = np.where(np.logical_not(binary_mask)[:, :, None], masked_image_full, np.array([255, 255, 255], dtype=masked_image_full.dtype))

        Image.fromarray(masked_image_full.astype(np.uint8)).save(masked_image_full_save_path)

        rle = single_mask_to_rle(binary_mask)
        result={
            'bbox': input_boxes[0].tolist(),
            'segmentation': rle,
        }

        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)

    with open(os.path.join(datasave, 'error', f'error_{rank}.txt'), 'w') as f:
        for error_file in error_files:
            f.write(error_file + '\n')
