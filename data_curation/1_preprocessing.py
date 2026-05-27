"""Step 1: Florence2 + GroundingDINO + GPT-4o entity extraction.

This script processes raw videos by:
1. Extracting a keyframe from each video
2. Generating captions using Florence-2
3. Extracting object nouns using GPT-4o
4. Detecting objects with GroundingDINO
5. Generating segmentation masks with SAM2
6. Saving results as JSON annotations

Usage:
    torchrun --nproc_per_node=NUM_GPUS 1_preprocessing.py \
        --p_meta metadata.csv --dataroot /path/to/videos --datasave /path/to/output
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

from utils.openai_client import get_openai_client
from datetime import timedelta
import openai


client = get_openai_client()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_meta', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--datasave', type=str, required=True)
    parser.add_argument('--n_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--copy_videoid', default=False, action='store_true')
    parser.add_argument('--sam2_checkpoint', type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--sam2_model_config', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--gdino_config', type=str, default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--gdino_checkpoint', type=str, default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument('--model_name', type=str, default="gpt4o", help="OpenAI model name for entity extraction")
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

def check_exist(datasave, filename, rank, args):
    if os.path.exists(os.path.join(datasave, filename[:-4] + '.json')):
        return True
    error_dir = os.path.join(datasave, 'error')
    try:
        with open(os.path.join(error_dir, f'{args.chunk_idx}_{args.n_chunks}_rank{rank}.txt'), 'r') as f:
            error_files = f.readlines()
        if filename + '\n' in error_files:
            return True
    except:
        pass
    return False

def find_start_index(meta, datasave, dataroot, rank, args):
    left, right = 0, len(meta) // dist.get_world_size() - 1
    adjust_flag = False
    while left < right:
        if not adjust_flag:
            mid = (left + right) // 2
        adjust_flag = False
        idx = mid * dist.get_world_size() + rank
        row = meta.iloc[idx]
        videoid = row['videoid']
        if str(videoid)[-4:] in ['.mp4', '.mov', '.MP4', '.MOV']:
            p_video = os.path.join(dataroot, str(videoid))
        else:
            p_video = os.path.join(dataroot, str(videoid) + '.mp4')

        if not os.path.exists(p_video):
            mid = mid + 1
            adjust_flag = True
            continue
        if args.copy_videoid:
            filename = videoid
        else:
            filename = os.path.basename(p_video)
        if check_exist(datasave, filename, rank, args):
            left = mid + 1
        else:
            right = mid
    idx = left * dist.get_world_size() + rank
    row = meta.iloc[idx]
    videoid = row['videoid']
    if str(videoid)[-4:] in ['.mp4', '.mov', '.MP4', '.MOV']:
        p_video = os.path.join(dataroot, str(videoid))
    else:
        p_video = os.path.join(dataroot, str(videoid) + '.mp4')
    if args.copy_videoid:
        filename = videoid
    else:
        filename = os.path.basename(p_video)
    if not check_exist(datasave, filename, rank, args):
        print(f"$$$$Bisection: last file not found, index is {left}, file is {filename}, path is {p_video}")
        return left
    else:
        print('$$$$Bisection: FINAL File already exists. Path: {}'.format(os.path.join(datasave, filename)))
        return -1

def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def process_single_text(original_caption, florence_caption, model_name="gpt4o"):
    content = f"Your role is to extract object nouns from the given text.\n\
                    From the given text, extract ALL objects nouns, and DO NOT include nouns that may refer to background.\n\
                    Remember we treat human as a kind of object here.\n\
                    Be aware of the context and make sure you really extracted a noun according to the context.\n\
                    The objects would be used for referring the object inside object segmentation model.\n\
                    Try to use the word inside the given sentence to extract the object nouns.\n\
                    Give me object name only, without any further context or description.\n\
                    The object nouns should be separated by a period '.'.\n\
                    Two types of caption will be provided, if there is object in both of the captions or mentioned in the context with another word, you only extract them once.\n\
                    Remember to ignore object with with excessive or unknown quantity (like 'many people'). Only record objects with a count of 3 or fewer.\n\
                    if no object words is found, please reply 'None' only.\n\
                    Please think carefully and follow every instruction.\n\
                    Before you submit your answer, please check if the object nouns are correct and complete. Filter any nouns that refers to the background.\n\
                    caption1 : {original_caption}\n\
                    caption2 : {florence_caption}"
    content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
    messages = [{
        "role": "user",
        "content": content
    }]
    retry = 0
    max_retries = 10
    base_delay = 2
    max_delay = 120
    while retry < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
                top_p=0.2,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            choice = response.choices[0].message.content
            return choice
        except (openai.AuthenticationError, openai.RateLimitError) as e:
            print(e)
            retry += 1
            delay = min(base_delay * (2 ** retry), max_delay)
            delay += random.uniform(0, 1)
            print(f"failed, wait for {delay:.2f} seconds... ( {retry}/{max_retries})")
            time.sleep(delay)
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing text: {original_caption} - {e}")
            return 'None'
    print(f"Failed to process text {original_caption} | after {max_retries} retries")
    return 'None'

def extract_object(original_caption, florence_caption, model_name="gpt4o"):
    object_nouns = process_single_text(original_caption, florence_caption, model_name)
    if object_nouns.lower() == 'none':
        return 'none'
    if not object_nouns.endswith('.'):
        object_nouns += '.'
    return object_nouns



if __name__ == '__main__':
    args = parse_args()
    p_meta = args.p_meta
    dataroot = args.dataroot
    datasave = args.datasave
    verbose = args.verbose
    os.makedirs(datasave, exist_ok=True)
    os.makedirs(os.path.join(datasave, 'error'), exist_ok=True)

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

    FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"
    florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
    florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)
    task_prompt = '<MORE_DETAILED_CAPTION>'

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


    meta = pd.read_csv(
        p_meta,
        on_bad_lines="skip",
        encoding="ISO-8859-1",
        engine="python",
        sep=",",
    )
    print('Total number of videos:', len(meta))

    # chunk the meta data for parallel processing
    assert args.n_chunks > 0, 'n_chunks must be greater than 0'
    if args.n_chunks > 1:
        n_chunks, chunk_idx = args.n_chunks, args.chunk_idx
        chunk_intervals = np.linspace(0, len(meta), n_chunks+1, dtype=int)
        chunk_metas = [meta.iloc[chunk_intervals[i]:chunk_intervals[i+1]]
                       for i in range(n_chunks)]
        meta = chunk_metas[chunk_idx]
        print('Processing chunk [{}/{}]'.format(chunk_idx, n_chunks))
        print('Number of videos in chunk:', len(meta))

    # Drop last samples so that can be divided via meta data
    n_videos = len(meta)
    n_videos = n_videos - n_videos % dist.get_world_size()
    meta = meta.iloc[:n_videos]
    start_index = find_start_index(meta, datasave, dataroot, rank, args)
    print(f'Start index: {start_index}')

    if start_index == -1:
        print(f'All videos have been processed, exit.')
        time.sleep(100)
    else:
        print(f'rank:{rank}, start_index: {start_index}')
        print('Number of videos after dropping:', len(meta))
        n_iterations = len(meta) // dist.get_world_size()
        try:
            for ite in tqdm.tqdm(range(start_index, n_iterations)):
                try:
                    if verbose:
                        start = time.time()
                    idx = ite * dist.get_world_size() + rank
                    if idx >= len(meta):
                        break
                    row = meta.iloc[idx]
                    video_id = row['videoid']
                    if args.copy_videoid:
                        video_path = os.path.join(dataroot, str(video_id))
                        filename = os.path.basename(video_id)[:-4]
                    else:
                        video_path = os.path.join(dataroot, str(video_id) + '.mp4')
                        filename = os.path.basename(video_id)
                    if not os.path.exists(video_path):
                        print(f'Video {video_id} does not exist, skip.')
                        error_dir = os.path.join(datasave, 'error')
                        os.makedirs(error_dir, exist_ok=True)
                        with open(os.path.join(error_dir, f"{args.chunk_idx}_{args.n_chunks}_rank{rank}.txt"), "a") as f:
                            f.write(f"{video_id}\n")
                        continue
                    vr = decord.VideoReader(video_path)
                    n_frames = len(vr)
                    mid_frame = n_frames // 2
                    keyframe = vr[mid_frame]
                    keyframe = Image.fromarray(keyframe.asnumpy()).convert("RGB")

                    if verbose:
                        end = time.time()
                        print(f'#####Video {video_id} loaded, time: {end-start:.2f}s')
                        start = time.time()
                    caption_results = run_florence2(task_prompt, None, florence2_model, florence2_processor, keyframe)
                    florence_caption = caption_results[task_prompt]
                    if verbose:
                        end = time.time()
                        print(f'#####Florence-2 caption generated, time: {end-start:.2f}s')
                        start = time.time()
                    original_caption = row['name']
                    object_nouns = extract_object(original_caption, florence_caption, args.model_name)
                    if object_nouns.lower() == 'none':
                        print(f'No object nouns found in video {video_id}, skip. caption is {original_caption}')
                        error_dir = os.path.join(datasave, 'error')
                        os.makedirs(error_dir, exist_ok=True)
                        with open(os.path.join(error_dir, f"{args.chunk_idx}_{args.n_chunks}_rank{rank}.txt"), "a") as f:
                            f.write(f"{video_id}\n")
                        continue
                    if verbose:
                        end = time.time()
                        print(f'#####Object nouns extracted, time: {end-start:.2f}s')
                        start = time.time()

                    image_source, image = load_image(keyframe)
                    sam2_predictor.set_image(image_source)
                    boxes, confidences, labels = predict(
                        model=grounding_model,
                        image=image,
                        caption=object_nouns,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD,
                    )
                    h, w, _ = image_source.shape
                    boxes = boxes * torch.Tensor([w, h, w, h])
                    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                    if verbose:
                        end = time.time()
                        print(f'#####Grounding-DINO model inference, time: {end-start:.2f}s')
                        start = time.time()

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, logits = sam2_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )

                    if verbose:
                        end = time.time()
                        print(f'#####SAM2 model inference, time: {end-start:.2f}s')
                        start = time.time()

                    if masks.ndim == 4:
                        masks = masks.squeeze(1)


                    confidences = confidences.numpy().tolist()
                    class_names = labels

                    class_ids = np.array(list(range(len(class_names))))

                    labels = [
                        f"{class_name} {confidence:.2f}"
                        for class_name, confidence
                        in zip(class_names, confidences)
                    ]

                    img_annotate = np.array(keyframe)
                    img_annotate = img_annotate[:, :, ::-1]
                    detections = sv.Detections(
                        xyxy=input_boxes,  # (n, 4)
                        mask=masks.astype(bool),  # (n, h, w)
                        class_id=class_ids
                    )
                    if args.copy_videoid:
                        output_dir = os.path.join(datasave, os.path.dirname(video_id))
                    else:
                        output_dir = os.path.join(datasave, os.path.dirname(video_id))
                        output_dir = os.path.join(output_dir, filename)
                    os.makedirs(output_dir, exist_ok=True)

                    box_annotator = sv.BoxAnnotator()
                    annotated_frame = box_annotator.annotate(scene=img_annotate.copy(), detections=detections)

                    label_annotator = sv.LabelAnnotator()
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

                    mask_annotator = sv.MaskAnnotator()
                    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                    cv2.imwrite(os.path.join(output_dir, f"{filename}.jpg"), annotated_frame)
                    keyframe.save(os.path.join(output_dir, f"original.png"))

                    # convert mask into rle format
                    mask_rles = [single_mask_to_rle(mask) for mask in masks]

                    input_boxes = input_boxes.tolist()
                    scores = scores.tolist()
                    # save the results in standard format
                    results = {
                        "video_path": video_id,
                        "keyframe_index": mid_frame,
                        "original_caption": original_caption,
                        "florence_caption": florence_caption,
                        "object_nouns": object_nouns,
                        "annotations" : [
                            {
                                "class_name": class_name,
                                "bbox": box,
                                "segmentation": mask_rle,
                                "score": score,
                            }
                            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
                        ],
                        "box_format": "xyxy",
                    }

                    with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
                        json.dump(results, f, indent=4)
                    if verbose:
                        end = time.time()
                        print(f'#####Results saved, time: {end-start:.2f}s')
                except Exception as e:
                    print(f"Error processing video {video_id}")
                    traceback.print_exc()
                    print('~~~~end traceback')
                    error_dir = os.path.join(datasave, 'error')
                    os.makedirs(error_dir, exist_ok=True)
                    with open(os.path.join(error_dir, f"{args.chunk_idx}_{args.n_chunks}_rank{rank}.txt"), "a") as f:
                        f.write(f"{video_id}\n")
                    continue
        except Exception as e:
            print(f'Unexpected error: {e}')
            traceback.print_exc()
    print(f'rank {rank} finished.')
    time.sleep(600)
    print(f"Rank {rank} is about to call dist.barrier()")
    dist.barrier()
    print(f"Rank {rank} has passed dist.barrier()")
