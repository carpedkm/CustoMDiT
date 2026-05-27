"""Step 3: GPT-4o quality/relevance filtering.

This script filters preprocessing annotations by:
1. Removing human body part annotations
2. Removing redundant class annotations (>3 instances)
3. Using GPT-4o to classify foreground vs background objects
4. Filtering by bounding box size (too small/too large)
5. Filtering by SAM confidence score
6. Removing repetitive overlapping detections (IoU-based NMS)

Usage:
    python 3_filtering.py --p_meta metadata.csv --dataroot /path/to/preprocessing_output \
        --datasave /path/to/output --n_chunks 16 --chunk_idx 0
"""

import os
import torch
import argparse
import numpy as np
import json
import pandas as pd
import tqdm
import time
import traceback
import random
import pycocotools.mask as mask_util
import re

from utils.openai_client import get_openai_client
import openai

client = None

def _get_client():
    global client
    if client is None:
        client = get_openai_client()
    return client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_meta', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--datasave', type=str, required=True)
    parser.add_argument('--n_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--copy_videoid', default=False, action='store_true')
    parser.add_argument('--model_name', type=str, default="gpt4o", help="OpenAI model name")
    args = parser.parse_args()
    return args

def check_exist(datasave, filename, args):
    if os.path.exists(os.path.join(datasave, filename[:-4] + '.json')):
        return True
    error_dir = os.path.join(datasave, 'error')
    try:
        with open(os.path.join(error_dir, f'{args.chunk_idx}_{args.n_chunks}.txt'), 'r') as f:
            error_files = f.readlines()
        if filename + '\n' in error_files:
            return True
    except:
        pass
    return False

def find_start_index(meta, datasave, dataroot, args):
    left, right = 0, len(meta)
    adjust_flag = False
    while left < right:
        if not adjust_flag:
            mid = (left + right) // 2
        adjust_flag = False
        idx = mid
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
        if check_exist(datasave, filename, args):
            left = mid + 1
        else:
            right = mid
    idx = left
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
    if not check_exist(datasave, filename, args):
        return left
    else:
        print('$$$$Bisection: FINAL File already exists. Path: {}'.format(os.path.join(datasave, filename)))
        return -1

def clean_annotations_remove_redundant(data, threshold=3):
    class_counts = {}
    for annotation in data.get("annotations", []):
        class_name = annotation["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    classes_to_remove = {class_name for class_name, count in class_counts.items() if count > threshold}

    filtered_annotations = [
        annotation for annotation in data.get("annotations", [])
        if annotation["class_name"] not in classes_to_remove
    ]

    data_loss = len(data["annotations"]) - len(filtered_annotations)

    data["annotations"] = filtered_annotations
    return data, data_loss

def process_single_text(class_name_str, model_name="gpt4o"):
    content = f"Your role is to differentiate between foreground words and background words in the given words.\n\
            The words are divided by '|'.\n\
            Please select the words that are foreground words and the words that are background words.\n\
            The background words refers to the background of a scene, for example, sky, room, ocean, forest.\n\
            Give me result only, without any further context or description.\n\
            Your output format should be in the following format:\n\
            foreground words: word1, word2, word3\n\
            background words: word4, word5, word6\n\
            Here are the words: {class_name_str}\
"
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
            response = _get_client().chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                top_p=0.95,
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
            print(f"Error processing text: {class_name_str} - {e}")
            return 'None'
    print(f"Failed to process text {class_name_str} | after {max_retries} retries")
    return 'None'

def clean_annotations_background(data, model_name="gpt4o"):
    class_names = {annotation["class_name"] for annotation in data.get("annotations", [])}
    class_names_str = "|".join(class_names)

    text = process_single_text(class_names_str, model_name)
    if text.lower() != "none":

        foreground_pattern = r"foreground words:\s*([^\n]+)"
        background_pattern = r"background words:\s*([^\n]+)"

        foreground_match = re.search(foreground_pattern, text, re.IGNORECASE)
        if foreground_match:
            foreground_words = [word.strip().lower() for word in foreground_match.group(1).split(',')]
        else:
            foreground_words = []

        background_match = re.search(background_pattern, text, re.IGNORECASE)
        if background_match:
            background_words = [word.strip().lower() for word in background_match.group(1).split(',')]
        else:
            background_words = []
        if len(background_words) != 0:
            filtered_annotations = [annotation for annotation in data.get("annotations", []) if annotation["class_name"].lower() not in background_words]
        else:
            filtered_annotations = data.get("annotations", [])
        data_loss = len(data["annotations"]) - len(filtered_annotations)

        data["annotations"] = filtered_annotations
    else:
        print("No valid response from OpenAI")
        data_loss = 0
    return data, data_loss

def clean_annotations_size(data, threshold_small=0.01, threshold_large=0.97, threshold_mask=0.8):
    filtered_annotations = []
    for annotation in data.get("annotations", []):
        bbox = annotation['bbox']
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        hp, wp = annotation['segmentation']['size']
        bbox_ratio = (w * h) / (wp * hp)
        if bbox_ratio >= threshold_small:
            if bbox_ratio <= threshold_large:
                filtered_annotations.append(annotation)
            else:
                segmentation = annotation['segmentation']
                binary_mask = mask_util.decode(segmentation)
                mask_area = np.sum(binary_mask)
                image_area = segmentation["size"][0] * segmentation["size"][1]
                percentage = mask_area / image_area
                if percentage <= threshold_mask:
                    filtered_annotations.append(annotation)
    data_loss = len(data["annotations"]) - len(filtered_annotations)

    data["annotations"] = filtered_annotations
    return data, data_loss

def clean_annotations_sam(data, threshold_sam=0.8):
    filtered_annotations = []
    for annotation in data.get("annotations", []):
        sam_score = annotation['score']
        if isinstance(sam_score, list):
            sam_score = sam_score[0]
        if sam_score >= threshold_sam:
            filtered_annotations.append(annotation)
    data_loss = len(data["annotations"]) - len(filtered_annotations)

    data["annotations"] = filtered_annotations
    return data, data_loss

def calculate_iou_upper_triangle(bboxes):
    bboxes = np.array(bboxes)

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    num_bboxes = len(bboxes)
    iou_matrix = np.zeros((num_bboxes, num_bboxes))

    for i in range(num_bboxes):
        x1, y1, x2, y2 = bboxes[i]

        xi1 = np.maximum(x1, bboxes[i + 1:, 0])
        yi1 = np.maximum(y1, bboxes[i + 1:, 1])
        xi2 = np.minimum(x2, bboxes[i + 1:, 2])
        yi2 = np.minimum(y2, bboxes[i + 1:, 3])

        inter_width = np.maximum(0, xi2 - xi1)
        inter_height = np.maximum(0, yi2 - yi1)

        intersection_area = inter_width * inter_height

        union_area = areas[i] + areas[i + 1:] - intersection_area

        iou_matrix[i, i + 1:] = intersection_area / union_area

    return iou_matrix

def clean_annotations_repetitive(data, threshold_iou=0.8):
    bboxes = []
    scores = []

    for annotation in data.get("annotations", []):
        bboxes.append(annotation['bbox'])
        scores.append(annotation['score'])

    iou_matrix = calculate_iou_upper_triangle(bboxes)
    num_bboxes = len(bboxes)
    to_keep = set(range(num_bboxes))

    for i in range(num_bboxes):
        if i not in to_keep:
            continue

        overlaps = np.where(iou_matrix[i, i + 1:] > threshold_iou)[0] + (i + 1)
        candidates = [i] + overlaps.tolist()

        best_idx = max(candidates, key=lambda idx: scores[idx])

        to_keep -= set(candidates)
        to_keep.add(best_idx)

    filtered_annotations = [data['annotations'][idx] for idx in to_keep]

    data_loss = len(data["annotations"]) - len(filtered_annotations)

    data["annotations"] = filtered_annotations
    return data, data_loss

def clean_annotations_remove_human(data):
    data_loss_human = 0
    annotations = data['annotations']
    annotations_filtered = []
    human_body_words = ['body', 'hand', 'hands', 'foot', 'feet', 'leg', 'legs', 'arm',
                        'arms', 'head', 'face', 'eyes', 'ear', 'ears', 'nose', 'neck',
                        'tongue', 'throat', 'mouth', 'lip', 'teeth', 'gums', 'tongue',
                        'fingers', 'toe', 'toes', 'hair', 'beard','eyebrow', 'eyelash',
                        'cheek', 'forehead', 'chin', 'jaw','shoulder', 'elbow', 'wrist',
                        'knuckle', 'thumb', 'fingernail','waist', 'hip', 'belly button',
                        'navel','knee', 'ankle', 'shin', 'thigh', 'toenail', 'skin']
    for annotation in annotations:
        if annotation['class_name'] not in human_body_words:
            annotations_filtered.append(annotation)
        else:
            data_loss_human += 1
    data['annotations'] = annotations_filtered
    return data, data_loss_human

if __name__ == '__main__':
    args = parse_args()
    p_meta = args.p_meta
    datasave = args.datasave
    dataroot = args.dataroot
    verbose = args.verbose
    os.makedirs(datasave, exist_ok=True)
    os.makedirs(os.path.join(datasave, 'data_loss'), exist_ok=True)
    os.makedirs(os.path.join(datasave, 'error'), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = pd.read_csv(
        p_meta,
        on_bad_lines="skip",
        encoding="ISO-8859-1",
        engine="python",
        sep=",",
    )
    print('Total number of videos:', len(meta))
    assert args.n_chunks > 0, 'n_chunks must be greater than 0'
    n_chunks, chunk_idx = args.n_chunks, args.chunk_idx
    if args.n_chunks > 1:
        chunk_intervals = np.linspace(0, len(meta), n_chunks+1, dtype=int)
        chunk_metas = [meta.iloc[chunk_intervals[i]:chunk_intervals[i+1]]
                       for i in range(n_chunks)]
        meta = chunk_metas[chunk_idx]
        print('Processing chunk [{}/{}]'.format(chunk_idx, n_chunks))
        print('Number of videos in chunk:', len(meta))

    # Drop last samples so that can be divided via meta data
    n_videos = len(meta)
    meta = meta.iloc[:n_videos]
    start_index = find_start_index(meta, datasave, dataroot, args)
    print(f'Start index: {start_index}')
    if start_index == -1:
        print(f'chunk_idx:{chunk_idx}, All videos have been processed, exit.')
        exit()

    meta = meta.iloc[start_index:]
    print('Number of videos to process:', len(meta))

    data_losses_redundant = []
    data_losses_background = []
    data_losses_size = []
    data_losses_sam = []
    data_losses_repetitive = []
    data_losses_human = []

    for idx in tqdm.tqdm(range(len(meta))):
        try:
            if verbose:
                start = time.time()
            row = meta.iloc[idx]
            video_id = row['videoid']
            if args.copy_videoid:
                json_path = os.path.join(dataroot, str(video_id)[:-4] + '.json')
                filename = os.path.basename(video_id)[:-4]
            else:
                filename = os.path.basename(video_id)
                dir_name = os.path.dirname(video_id)
                json_path = os.path.join(dataroot, dir_name, filename, str(filename) + '.json')
            if not os.path.exists(json_path):
                print(f'json {json_path} does not exist, skip.')
                continue

            # read json file
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Remove human body part annotations
            data, data_loss_human = clean_annotations_remove_human(data)
            if len(data['annotations']) == 0:
                print(f'video {video_id} has no annotations after removing human body annotations, skip.')
                with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                    f.write(f'{video_id}\n')
                continue

            # step 1. Remove redundant class annotations
            data, data_loss_redundant = clean_annotations_remove_redundant(data)

            if len(data['annotations']) == 0:
                print(f'video {video_id} has no annotations after removing redundant class annotations, skip.')
                with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                    f.write(f'{video_id}\n')
                continue

            # step 2. Filter out words refering to the background
            data, data_loss_background = clean_annotations_background(data, args.model_name)

            if len(data['annotations']) == 0:
                print(f'video {video_id} has no annotations after filtering out background words, skip.')
                with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                    f.write(f'{video_id}\n')
                continue

            # step 3. Remove identity that is too small or large
            data, data_loss_size = clean_annotations_size(data)

            if len(data['annotations']) == 0:
                print(f'video {video_id} has no annotations after filtering out small or large objects, skip.')
                with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                    f.write(f'{video_id}\n')
                continue

            # step 4. Remove identity with low SAM score
            data, data_loss_sam = clean_annotations_sam(data)

            if len(data['annotations']) == 0:
                print(f'video {video_id} has no annotations after filtering out low SAM score, skip.')
                with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                    f.write(f'{video_id}\n')
                continue

            #step 5. Remove repetitive objects
            data, data_loss_repetitive = clean_annotations_repetitive(data)

            if len(data['annotations']) == 0:
                print(f'video {video_id} has no annotations after filtering out repetitive objects, skip.')
                with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                    f.write(f'{video_id}\n')
                continue

            data_losses_redundant.append(data_loss_redundant)
            data_losses_background.append(data_loss_background)
            data_losses_size.append(data_loss_size)
            data_losses_sam.append(data_loss_sam)
            data_losses_repetitive.append(data_loss_repetitive)
            data_losses_human.append(data_loss_human)

            # save data
            if args.copy_videoid:
                output_dir = os.path.join(datasave, os.path.dirname(video_id))
            else:
                output_dir = os.path.join(datasave, os.path.dirname(video_id))
                output_dir = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f'{filename}.json'), 'w') as f:
                json.dump(data, f, indent=4)

        except Exception as e:
            print(f"Error processing video {video_id}")
            traceback.print_exc()
            print('~~~~end traceback')
            continue

    # save data loss array
    data_losses_redundant = np.array(data_losses_redundant)
    data_losses_background = np.array(data_losses_background)
    data_losses_size = np.array(data_losses_size)
    data_losses_sam = np.array(data_losses_sam)
    data_losses_repetitive = np.array(data_losses_repetitive)
    data_losses_human = np.array(data_losses_human)
    data_losses = np.stack([data_losses_redundant, data_losses_background, data_losses_size, data_losses_sam, data_losses_repetitive, data_losses_human], axis=0)
    print(f'total data loss: {np.sum(data_losses)}')
    np.save(os.path.join(datasave, 'data_loss', f'data_losses_{n_chunks}_{chunk_idx}.npy'), data_losses)
