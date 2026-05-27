"""Step 4: GPT-4o subject-centric re-captioning.

This script generates object-centric captions for each detected subject by:
1. Re-selecting class names to match original/Florence captions
2. Sending full image + cropped object image to GPT-4o for focused captioning
3. Batch-processing unique objects and individual processing for duplicates
4. Adding a special token 'loicea' before the subject word in each caption

Usage:
    python 4_recaptioning.py --p_meta metadata.csv --dataroot /path/to/filtered_output \
        --datasave /path/to/output --image_root /path/to/preprocessing_output
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
from PIL import Image
import io
import base64
from collections import Counter, defaultdict

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
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--n_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--copy_videoid', default=False, action='store_true')
    parser.add_argument('--model_name', type=str, default="gpt4o", help="OpenAI model name")
    args = parser.parse_args()
    return args

def check_exist(datasave, filename, args):
    if args.copy_videoid:
        if os.path.exists(os.path.join(datasave, filename[:-4] + '.json')):
            return True
    else:
        filename = filename[:-4]
        if os.path.exists(os.path.join(datasave, filename, filename + '.json')):
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

def annotation_reselecting(data):
    annotations = data.get("annotations", [])
    original_caption = data.get("original_caption", "").lower()
    florence_caption = data.get("florence_caption", "").lower()
    filtered_annotations = []
    for annotation in annotations:
        class_name = annotation["class_name"]
        if class_name.lower() not in original_caption and class_name.lower() not in florence_caption:
            class_name_candidates = class_name.strip().split(' ')
            if len(class_name_candidates) <= 1:
                continue
            class_name_candidates = class_name_candidates[::-1]
            for class_name_candidate in class_name_candidates:
                if class_name_candidate.lower() in original_caption or class_name_candidate.lower() in florence_caption:
                    annotation["class_name"] = class_name_candidate
                    filtered_annotations.append(annotation)
                    break
        else:
            filtered_annotations.append(annotation)
    data["annotations"] = filtered_annotations
    return data

def encode_image(image_path, crop=None, size_max=1024):
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        if original_width > size_max or original_height > size_max:
            aspect_ratio = original_width / original_height
            if original_width > original_height:
                new_width = size_max
                new_height = round(new_width / aspect_ratio)
            else:
                new_height = size_max
                new_width = round(new_height * aspect_ratio)
            img = img.resize((new_width, new_height))
        else:
            new_width, new_height = original_width, original_height
        if crop is not None:
            scale_x = new_width / original_width
            scale_y = new_height / original_height

            x1 = crop[0] * scale_x
            y1 = crop[1] * scale_y
            x2 = crop[2] * scale_x
            y2 = crop[3] * scale_y

            cropped_image = img.crop((x1, y1, x2, y2)).copy()

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        if crop is not None:
            buffer = io.BytesIO()
            cropped_image.save(buffer, format="JPEG")
            crop_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_b64, crop_img_b64
        else:
            return img_b64

def process_batch_class(class_name, image_path, original_caption, model_name="gpt4o", verbose=False):
    if verbose:
        start = time.time()
    content = f"Your role is to recaption the given image following instructions.\n\
            The caption should be focus on the given object.\n\
            You should generate the same number of caption corresponding to the number of objects provided. Each caption is independent. They will be used separately.\n\
            The objects are divided with '|' symbol.\n\
            The caption MUST include the given object words without modifying.\n\
            Try to describe interactions between different objects in the scene.\n\
            In each line reply the caption only, without any further context or description.\n\
            Remember to add the word 'loicea' only before the given object words correctly.\n\
            'loicea' should appear in the caption once and only once in each caption.\n\
            For reference only, the original caption is {original_caption}, you might merge it to your caption, but you don't have to.\n\
            The given objects are \"{class_name}\""
    content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
    img_b64 = encode_image(image_path, size_max=1024)
    messages = [
    {
        "role": "user",
        "content": [
          {
            'type': 'text',
            'text': content
          },
          {
            'type': 'image_url',
            'image_url': {
              'url': f'data:image/jpeg;base64,{img_b64}',
              'detail': 'high'
          }
          },
        ]
    }
    ]
    retry = 0
    policy_retry = False
    max_retries = 10
    base_delay = 2
    max_delay = 120
    while retry < max_retries:
        try:
            response = _get_client().chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=2000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            choice = response.choices[0].message.content
            if verbose:
                print(f"Time taken process batch class: {time.time()-start:.2f} seconds")
            return choice
        except (openai.AuthenticationError, openai.RateLimitError) as e:
            print(e)
            retry += 1
            delay = min(base_delay * (2 ** retry), max_delay)
            delay += random.uniform(0, 1)
            print(f"failed, wait for {delay:.2f} seconds... ( {retry}/{max_retries})")
            time.sleep(delay)
        except openai.BadRequestError as e:
            if policy_retry:
                print('policy retry triggered before.')
                return 'None'
            policy_retry = True
            print(e)
            content += '\nThe image could not be provided for some reason, please give your response only according to original caption, following the instructions.'
            messages = [
            {
                "role": "user",
                "content": [
                {
                    'type': 'text',
                    'text': content
                },
                ]
            }
            ]
            retry += 1
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing text: {image_path} - {e}")
            return 'None'
    print(f"Failed to process text {image_path} | after {max_retries} retries")
    return 'None'


def process_single_text(class_name, image_path, original_caption, bbox, model_name="gpt4o", verbose=False):
    if verbose:
        start = time.time()
    content = f"Your role is to recaption the given image following instructions.\n\
            The caption should be focus on the given object.\n\
            The caption MUST include the given object words without modifying.\n\
            Try to describe interactions between different objects in the scene.\n\
            Reply the caption only, without any further context or description.\n\
            There might be several objects that match the given object name, so you can refer to the second image to choose the correct object.\n\
            Remember to add the word 'loicea' only before the given object words in the second image.\n\
            'loicea' should appear in the caption once and only once.\n\
            For reference only, the original caption is {original_caption}.\n\
            The given object is \"{class_name}\""
    content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
    img_b64, crop_img_b64 = encode_image(image_path, crop=bbox, size_max=2048)
    messages = [
    {
        "role": "user",
        "content": [
          {
            'type': 'text',
            'text': content
          },
          {
            'type': 'image_url',
            'image_url': {
              'url': f'data:image/jpeg;base64,{img_b64}',
              'detail': 'high'
          }
          },
          {
            'type': 'image_url',
            'image_url': {
              'url': f'data:image/jpeg;base64,{crop_img_b64}',
              'detail': 'high'
          }
          }
        ]
    }
    ]
    retry = 0
    policy_retry = False
    max_retries = 10
    base_delay = 2
    max_delay = 120
    while retry < max_retries:
        try:
            response = _get_client().chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=2000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            choice = response.choices[0].message.content
            if verbose:
                end = time.time()
                print(f"Time taken process single text: {end-start:.2f} seconds")
            return choice
        except (openai.AuthenticationError, openai.RateLimitError) as e:
            print(e)
            retry += 1
            delay = min(base_delay * (2 ** retry), max_delay)
            delay += random.uniform(0, 1)
            print(f"failed, wait for {delay:.2f} seconds... ( {retry}/{max_retries})")
            time.sleep(delay)
        except openai.BadRequestError as e:
            if policy_retry:
                print('policy retry triggered before.')
                return 'None'
            policy_retry = True
            print(e)
            content += '\nThe image could not be provided for some reason, please give your response only according to original caption, following the instructions.'
            messages = [
            {
                "role": "user",
                "content": [
                {
                    'type': 'text',
                    'text': content
                },
                ]
            }
            ]
            retry += 1
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing text: {image_path} - {e}")
            return 'None'
    print(f"Failed to process text {image_path} | after {max_retries} retries")
    return 'None'

def object_centric_recaptioning(data, args, filename):
    annotations = data.get("annotations", [])
    processed_annotations = []
    original_caption = data.get("original_caption", "")
    videoid_path = os.path.dirname(data['video_path'])
    if args.copy_videoid:
        image_path = os.path.join(args.image_root, videoid_path, 'original.png')
    else:
        image_path = os.path.join(args.image_root, videoid_path, filename, 'original.png')
    class_annotation = [annotation['class_name'] for annotation in annotations]
    counter = Counter(class_annotation)

    duplicates = {class_name for class_name, count in counter.items() if count > 1}
    annotations_by_class = defaultdict(list)
    annotation_single = []
    for annotation in annotations:
        if annotation['class_name'] in duplicates:
            annotations_by_class[annotation['class_name']].append(annotation)
        else:
            annotation_single.append(annotation)
    annotations_by_class = dict(annotations_by_class)

    if len(duplicates) != 0:
        print('processing multipled words not included')
        if random.random() < 0.1:
            print(f'words for 1/10 possibility: {duplicates}')
            print(f'video_path: {data["video_path"]}')
        for class_name, class_annotations in annotations_by_class.items():
            for annotation in class_annotations:
                class_name = annotation["class_name"]
                bbox = annotation['bbox']
                recaption_result = process_single_text(class_name, image_path, original_caption, bbox, args.model_name)
                if recaption_result.lower() != "none":
                    annotation['caption'] = recaption_result
                    processed_annotations.append(annotation)
                else:
                    print("No valid response from OpenAI")
    if len(annotation_single) != 0:
        class_names = [annotation['class_name'] for annotation in annotation_single]
        class_names = '|'.join(class_names)
        recaption_result = process_batch_class(class_names, image_path, original_caption, args.model_name, verbose=args.verbose)
        retry = 0
        while not recaption_result:
            retry += 1
            if retry > 3:
                break
            recaption_result = process_batch_class(class_names, image_path, original_caption, args.model_name, verbose=args.verbose)
        if not recaption_result:
            print("No valid response from OpenAI, skip")
            data['annotations'] = processed_annotations
            return data
        recaption_result_list = [line for line in recaption_result.split('\n') if line]
        if recaption_result.lower() != "none":
            for i, annotation in enumerate(annotation_single):
                try:
                    recaption = recaption_result_list[i]
                except Exception as e:
                    print(f"Error processing text while getting index: {image_path} - {e}")
                    print(f'###@@@\n{recaption_result}\n###@@@')
                if annotation['class_name'].lower() in recaption.lower():
                    annotation['caption'] = recaption
                    processed_annotations.append(annotation)
                else:
                    print("No valid response from OpenAI, maybe not in order")
                    print(f"class_name: {annotation['class_name']}, result: {recaption_result}")
        else:
            print("No valid response from OpenAI")
    data['annotations'] = processed_annotations
    return data

if __name__ == '__main__':
    args = parse_args()
    p_meta = args.p_meta
    datasave = args.datasave
    dataroot = args.dataroot
    verbose = args.verbose
    os.makedirs(datasave, exist_ok=True)
    os.makedirs(os.path.join(datasave, 'num_annotation'), exist_ok=True)
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

    n_videos = len(meta)
    meta = meta.iloc[:n_videos]
    start_index = find_start_index(meta, datasave, dataroot, args)
    print(f'Start index: {start_index}')
    if start_index == -1:
        print(f'chunk_idx:{chunk_idx}, All videos have been processed, exit.')
        exit()

    meta = meta.iloc[start_index:]
    print('Number of videos to process:', len(meta))

    num_annotation_dict = {}

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
            if verbose:
                print(f'time loading json: {time.time() - start:.2f}s')
                start = time.time()
            # step 1. Reselect class name for multipled words not included
            data = annotation_reselecting(data)

            if len(data['annotations']) == 0:
                print(f'video {video_id} has no annotations after annotation reselecting, skip.')
                with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                    f.write(f'{video_id}\n')
                continue
            if verbose:
                print(f'time reselecting: {time.time() - start:.2f}s')
                start = time.time()

            # step 2. Object-centric recaptioning
            data = object_centric_recaptioning(data, args, filename)

            if len(data['annotations']) == 0:
                print(f'video {video_id} has no annotations after object centric recaptioning, skip.')
                with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                    f.write(f'{video_id}\n')
                continue

            num_annotation = len(data['annotations'])
            num_annotation_dict[video_id] = num_annotation

            if verbose:
                print(f'time object centric recaptioning: {time.time() - start:.2f}s')
                start = time.time()

            # save data
            if args.copy_videoid:
                output_dir = os.path.join(datasave, os.path.dirname(video_id))
            else:
                output_dir = os.path.join(datasave, os.path.dirname(video_id))
                output_dir = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f'{filename}.json'), 'w') as f:
                json.dump(data, f, indent=4)

            if verbose:
                print(f'time saving json: {time.time() - start:.2f}s')
                start = time.time()

        except Exception as e:
            print(f"Error processing video {video_id}")
            traceback.print_exc()
            print('~~~~end traceback')
            with open(os.path.join(datasave, 'error', f'error_{n_chunks}_{chunk_idx}.txt'), 'a') as f:
                f.write(f'{video_id}\n')
            continue

    # save num_annotation_dict as json
    with open(os.path.join(datasave, 'num_annotation', f'{n_chunks}_{chunk_idx}.json'), 'w') as f:
        json.dump(num_annotation_dict, f, indent=4)
