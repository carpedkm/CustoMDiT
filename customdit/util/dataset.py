"""Dataset and dataloader for CustomDiT training on PexelsCustom-1M."""

import gc
import itertools
import json
import math
import os
import random
import traceback
from abc import abstractmethod
from functools import reduce
from operator import mul
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import decord
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import safetensors
import torch
import torch.distributed as dist
from PIL import Image
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms

from .util import resize_and_pad


decord.bridge.set_bridge("torch")


def augment_object(image, segmentation, bbox, fill=(127,127,127),
                    prob_scale=0.2, prob_translate=0.2, prob_rotate=0.2):
    """
    对图像中的目标区域（由 bbox 给出，格式为 [x1, y1, x2, y2]）进行依次随机的缩放、平移、旋转变换，
    并在变换前先利用 segmentation 得到二值 mask，将图像中非目标部分填充为指定颜色 fill。

    变换顺序固定为：先缩放，再平移，再旋转。每一步变换都随机触发（触发概率分别为 prob_scale, prob_translate, prob_rotate），
    同时保证变换后目标区域（由 bbox 四个顶点表示）完全位于图像内部。

    参数：
      image: PIL.Image 对象
      segmentation: 分割数据（符合 pycocotools.mask.decode 格式）
      bbox: [x1, y1, x2, y2]，目标区域的轴对齐边界框
      fill: 填充颜色，例如 (R, G, B)
      prob_scale: 缩放操作的触发概率
      prob_translate: 平移操作的触发概率
      prob_rotate: 旋转操作的触发概率

    返回：
      变换后的 PIL.Image 对象
    """
    # 1. 利用 segmentation 得到目标的二值 mask，并将非目标区域填充为 fill 颜色
    binary_mask = mask_util.decode(segmentation).astype(bool)
    image_np = np.array(image)
    image_np[~binary_mask] = fill
    im_h, im_w = image_np.shape[0], image_np.shape[1]

    # 用于后续操作的基础图像（以 PIL.Image 格式）
    base_img = Image.fromarray(image_np)

    # 2. 将 bbox 转换为四个顶点坐标 (顺序：左上、右上、右下、左下)
    x1, y1, x2, y2 = bbox
    wbox,hbox = x2-x1, y2-y1
    wbox,hbox = round(wbox), round(hbox)
    corners = np.array([[x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]], dtype=np.float32)
    # 当前区域中心
    center = np.array([(x1+x2)/2, (y1+y2)/2], dtype=np.float32)

    # ---------------------
    # 变换步骤 1：缩放
    # ---------------------
    if random.random() < prob_scale:
        half_w = center[0] - x1
        half_h = center[1] - y1
        if half_w <= 0 or half_h <= 0:
            scale_factor = 1.0
        else:
            max_scale = min(center[0] / half_w, (im_w - center[0]) / half_w,
                            center[1] / half_h, (im_h - center[1]) / half_h)
            min_scale = 0.8
            max_scale = min(max_scale, 1.5)
            if max_scale < min_scale:
                scale_factor = 1.0
            else:
                scale_factor = random.uniform(min_scale, max_scale)
        new_corners = center + scale_factor * (corners - center)
        new_x1, new_y1 = new_corners.min(axis=0).astype(int)
        new_x2, new_y2 = new_corners.max(axis=0).astype(int)

        region = base_img.crop((int(x1), int(y1), int(x2), int(y2)))
        new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
        scaled_region = region.resize((new_w, new_h), resample=Image.BICUBIC)

        fill_region = Image.new('RGB', (int(x2-x1), int(y2-y1)), fill)
        base_img.paste(fill_region, (int(x1), int(y1)))
        base_img.paste(scaled_region, (new_x1, new_y1))

        corners = new_corners
        x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
        center = np.array([(x1+x2)/2, (y1+y2)/2], dtype=np.float32)

    # ---------------------
    # 变换步骤 2：平移
    # ---------------------
    if random.random() < prob_translate:
        max_left = int(x1)
        max_left = min(max_left, wbox)
        max_right = int(im_w - x2)
        max_right = min(max_right, wbox)
        dx = random.randint(-max_left, max_right)
        max_up = int(y1)
        max_up = min(max_up, hbox)
        max_down = int(im_h - y2)
        max_down = min(max_down, hbox)
        dy = random.randint(-max_up, max_down)

        new_corners = corners + np.array([dx, dy], dtype=np.float32)
        new_x1, new_y1 = new_corners.min(axis=0).astype(int)
        new_x2, new_y2 = new_corners.max(axis=0).astype(int)

        region = base_img.crop((int(x1), int(y1), int(x2), int(y2)))
        fill_region = Image.new('RGB', (int(x2-x1), int(y2-y1)), fill)
        base_img.paste(fill_region, (int(x1), int(y1)))
        base_img.paste(region, (new_x1, new_y1))

        corners = new_corners
        x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
        center = np.array([(x1+x2)/2, (y1+y2)/2], dtype=np.float32)

    # ---------------------
    # 变换步骤 3：旋转
    # ---------------------
    if random.random() < prob_rotate:
        candidate_angles = list(range(-15, 16))
        valid_angles = []
        for angle in candidate_angles:
            theta = math.radians(angle)
            R = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta),  math.cos(theta)]])
            rotated = center + np.dot(corners - center, R.T)
            if (rotated[:,0].min() >= 0 and rotated[:,1].min() >= 0 and
                rotated[:,0].max() <= im_w and rotated[:,1].max() <= im_h):
                valid_angles.append(angle)
        if valid_angles:
            chosen_angle = random.choice(valid_angles)
        else:
            chosen_angle = 0
        theta = math.radians(chosen_angle)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        new_corners = center + np.dot(corners - center, R.T)
        new_x1, new_y1 = new_corners.min(axis=0).astype(int)
        new_x2, new_y2 = new_corners.max(axis=0).astype(int)

        region = base_img.crop((int(x1), int(y1), int(x2), int(y2)))
        rotated_region = region.rotate(chosen_angle, resample=Image.BICUBIC, expand=True, fillcolor=fill)
        new_region_w, new_region_h = rotated_region.size
        new_region_x = int(center[0] - new_region_w / 2)
        new_region_y = int(center[1] - new_region_h / 2)
        fill_region = Image.new('RGB', (int(x2-x1), int(y2-y1)), fill)
        base_img.paste(fill_region, (int(x1), int(y1)))
        base_img.paste(rotated_region, (new_region_x, new_region_y))

        corners = new_corners
        x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2

    return base_img


def init_transform_dict(
    input_res_h=224,
    input_res_w=224,
    randcrop_scale=(0.5, 1.0),
    color_jitter=(0, 0, 0),
    norm_mean=(0.5, 0.5, 0.5),
    norm_std=(0.5, 0.5, 0.5),
):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        "train": transforms.Compose(
            [
                transforms.Resize(input_res_h, antialias=True),
                transforms.CenterCrop((input_res_h, input_res_w)),
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_res_h, antialias=True),
                transforms.CenterCrop((input_res_h, input_res_w)),
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_res_h, antialias=True),
                transforms.CenterCrop((input_res_h, input_res_w)),
                normalize,
            ]
        ),
    }
    return tsfm_dict


def init_transform_dict_resizedcrop(
    input_res_h=224,
    input_res_w=224,
    randcrop_scale=(0.5, 1.0),
    color_jitter=(0, 0, 0),
    norm_mean=(0.5, 0.5, 0.5),
    norm_std=(0.5, 0.5, 0.5),
):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (input_res_h, input_res_w), scale=randcrop_scale, antialias=True
                ),
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_res_h, antialias=True),
                transforms.CenterCrop((input_res_h, input_res_w)),
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_res_h, antialias=True),
                transforms.CenterCrop((input_res_h, input_res_w)),
                normalize,
            ]
        ),
    }
    return tsfm_dict


def init_transform_dict_crop(
    input_res_h=224,
    input_res_w=224,
    randcrop_scale=(0.5, 1.0),
    color_jitter=(0, 0, 0),
    norm_mean=(0.5, 0.5, 0.5),
    norm_std=(0.5, 0.5, 0.5),
):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        "train": transforms.Compose(
            [
                transforms.RandomCrop((input_res_h, input_res_w)),
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.CenterCrop((input_res_h, input_res_w)),
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.CenterCrop((input_res_h, input_res_w)),
                normalize,
            ]
        ),
    }
    return tsfm_dict


def init_transform_dict_normalizeonly(
    input_res_h=224,
    input_res_w=224,
    randcrop_scale=(0.5, 1.0),
    color_jitter=(0, 0, 0),
    norm_mean=(0.5, 0.5, 0.5),
    norm_std=(0.5, 0.5, 0.5),
):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        "train": transforms.Compose(
            [
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                normalize,
            ]
        ),
    }
    return tsfm_dict


def my_resize_and_crop_optimized(x, min_block_size=64, max_pixel=512*512, min_pixel=320*320):
    assert min_pixel < max_pixel, 'min_pixel must be less than max_pixel'
    assert min_block_size**2 <= min_pixel, 'min_block_size must be less than or equal to min_pixel'
    assert len(x.shape) == 4, 'x must be 4D tensor'

    T, C, H, W = x.shape
    target_h, target_w = H, W

    if (H * W) > max_pixel:
        shrink_ratio = np.sqrt(max_pixel / (H * W))
        target_h = int(np.round(H * shrink_ratio))
        target_w = int(np.round(W * shrink_ratio))
        x = transforms.Resize((target_h, target_w), antialias=True)(x)
    elif (H * W) < min_pixel:
        expand_ratio = np.sqrt(min_pixel / (H * W))
        target_h = int(np.round(H * expand_ratio))
        target_w = int(np.round(W * expand_ratio))
        x = transforms.Resize((target_h, target_w), antialias=True)(x)

    T, C, H, W = x.shape
    h_length = H // min_block_size * min_block_size
    w_length = W // min_block_size * min_block_size

    h_start = (H - h_length) // 2
    w_start = (W - w_length) // 2
    x = x[:, :, h_start:h_start+h_length, w_start:w_start+w_length]

    return x


def sample_frames(
    num_frames, vlen, sample="rand", fix_start=None, **kwargs
):
    """
    num_frames: The number of frames to sample.
    vlen: The length of the video.
    sample: The sampling method.
        choices of frame_sample:
        - 'equally spaced': sample frames equally spaced
        - 'proportional': sample frames proportional to the length of the frames in one second
        - 'random': sample frames randomly (not recommended)
        - 'uniform': sample frames uniformly (not recommended)
    fix_start: The starting frame index. If it is not None, then it will be used as the starting frame index.
    """
    acc_samples = min(num_frames, vlen)
    if sample in ["rand", "uniform"]:
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif fix_start is not None:
            frame_idxs = [x[0] + fix_start for x in ranges]
        elif sample == "uniform":
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    elif sample in ["equally spaced", "proportional"]:
        if sample == "equally spaced":
            raise NotImplementedError
        else:
            interval = round(kwargs["fps"] / kwargs["sample_factor"])
            needed_frames = (acc_samples - 1) * interval

            if fix_start is not None:
                start = fix_start
            else:
                if vlen - needed_frames - 1 < 0:
                    start = 0
                else:
                    start = random.randint(0, vlen - needed_frames - 1)
            frame_idxs = np.linspace(
                start=start, stop=min(vlen - 1, start + needed_frames), num=acc_samples
            ).astype(int)
    elif sample == "middle":
        interval = round(kwargs["fps"] / kwargs["sample_factor"])
        needed_frames = (acc_samples - 1) * interval
        fix_start = max(0, vlen // 2 - needed_frames // 2)
        frame_idxs = np.linspace(
                            fix_start,
                            min(fix_start + needed_frames, vlen - 1),
                            acc_samples,
                            dtype=int)
    elif sample == "middle_random":
        try:
            interval = round(kwargs["fps"] / kwargs["sample_factor"])
            needed_frames = (acc_samples - 1) * interval
            mid = vlen // 2
            lower_bound = max(0, mid - needed_frames)
            upper_bound = min(mid, vlen - 1 - needed_frames)
            if lower_bound > upper_bound:
                start = lower_bound
            else:
                start = random.randint(lower_bound, upper_bound)
            frame_idxs = np.linspace(start, min(start + needed_frames, vlen - 1), acc_samples, dtype=int)
        except Exception as e:
            traceback.print_exc()
    elif sample == "middle_extend_random":
        try:
            interval = round(kwargs["fps"] / kwargs["sample_factor"])
            mid = vlen // 2
            needed_frames = (acc_samples - 1) * interval
            start_range = max(0, mid - int(1.5 * needed_frames))
            end_range = min(vlen - needed_frames, mid + int(0.5 * needed_frames))
            if start_range >= end_range:
                start = start_range
            else:
                start = random.randint(start_range, end_range)
            frame_idxs = np.linspace(start, min(start + needed_frames, vlen - 1), acc_samples, dtype=int)
        except Exception as e:
            traceback.print_exc()
    else:
        raise NotImplementedError
    return frame_idxs


def read_frames_cv2(video_path, num_frames, sample="rand", fix_start=None, **kwargs):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idxs = sample_frames(
        num_frames,
        vlen,
        sample=sample,
        fix_start=fix_start,
        fps=fps,
        sample_factor=kwargs["sample_factor"],
    )
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs


def read_frames_decord(video_path, num_frames, sample="rand", fix_start=None, **kwargs):
    max_longedge_of_load = kwargs.get("max_longedge_of_load", None)
    if max_longedge_of_load is not None:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        height, width, _ = vr[0].shape
        if max(height, width) > max_longedge_of_load:
            shrink_ratio = max_longedge_of_load / max(height, width)
            height, width = int(
                height * shrink_ratio), int(width * shrink_ratio)
        video_reader = decord.VideoReader(video_path, num_threads=0, width=width, height=height)
    else:
        video_reader = decord.VideoReader(video_path, num_threads=0)

    min_shortedge_of_load = kwargs.get("min_shortedge_of_load", None)
    if min_shortedge_of_load is not None:
        height, width, _ = video_reader[0].shape
        if min(height, width) < min_shortedge_of_load:
            shrink_ratio = min_shortedge_of_load / min(height, width)
            height, width = np.ceil(height * shrink_ratio).astype(int), np.ceil(width * shrink_ratio).astype(int)
        video_reader = decord.VideoReader(video_path, num_threads=0, width=width, height=height)

    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    frame_idxs = sample_frames(
        num_frames,
        vlen,
        sample=sample,
        fix_start=fix_start,
        fps=fps,
        sample_factor=kwargs["sample_factor"],
    )
    try:
        frames = video_reader.get_batch(frame_idxs)
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
    except Exception as e:
        traceback.print_exc()
    return frames, frame_idxs


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen


video_reader = {
    "cv2": read_frames_cv2,
    "decord": read_frames_decord,
}


class TextVideoDatasetPT2V(Dataset):
    def __init__(
        self,
        dataset_name,
        text_params,
        video_params,
        data_dir,
        metadata_dir=None,
        metadata_folder_name=None,
        split="train",
        tsfms=None,
        cut=None,
        key=None,
        subsample=1,
        sliding_window_stride=-1,
        reader="decord",
        first_stage_key="video",
        cond_stage_key="txt",
        skip_missing_files=True,
        latent_dirname=None,
        t5_latent_dir=None,
        use_t5_latent=False,
        cond_image_dir=None,
        cond_json_dir=None,
        metadata_path=None,
        data_augmentation=False,
        drop_mask=None,
        use_recaption=True,
        is_filter=False,
    ):
        self.is_filter = is_filter
        self.use_recaption = use_recaption
        self.drop_mask = drop_mask
        self.data_augmentation = data_augmentation
        self.metadata_path = metadata_path
        print(f'metadata_path: {self.metadata_path}')
        self.cond_image_dir = cond_image_dir
        self.cond_json_dir = cond_json_dir
        self.t5_latent_dir = t5_latent_dir
        self.use_t5_latent = use_t5_latent
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        self.data_dir = os.path.expandvars(data_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        self.metadata_folder_name = metadata_folder_name
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.skip = skip_missing_files
        self.lack_files = []
        self.split = split
        self.key = key
        tsfm_params = (
            {}
            if "tsfm_params" not in video_params.keys()
            else video_params["tsfm_params"]
        )
        tsfm_params["input_res_h"] = video_params["input_res_h"]
        tsfm_params["input_res_w"] = video_params["input_res_w"]
        tsfm_type = video_params.get("tsfm_type", "standard")
        if tsfm_type == "standard":
            tsfm_dict = init_transform_dict(**tsfm_params)
        elif tsfm_type == "crop":
            tsfm_dict = init_transform_dict_crop(**tsfm_params)
        elif tsfm_type == "resizedcrop":
            tsfm_dict = init_transform_dict_resizedcrop(**tsfm_params)
        else:
            raise ValueError(f"tsfm_type {tsfm_type} not recognized")

        if split not in ["train", "val", "test"]:
            print(
                'Warning: split is not in ["train", "val", "test"], '
                'what you set is "{}", '
                'set it to "train"'.format(split)
            )
            split = "train"

        tsfms = tsfm_dict[split]

        self.transforms = tsfms
        self.transforms_cond = transforms.Compose([
            transforms.Lambda(lambda img: resize_and_pad(img, target_width=video_params["input_res_w"], target_height=video_params["input_res_h"], fill=(127,127,127))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.condition_target_shape=(video_params["input_res_h"], video_params["input_res_w"])

        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.video_reader_max_longedge_of_load = video_params.get(
            "max_longedge_of_load", None
        )
        self.video_reader_min_shortedge_of_load = video_params.get(
            "min_shortedge_of_load", None
        )
        self.label_type = "caption"
        self.frame_sample = video_params.get("frame_sample", "proportional")
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != "test":
                raise ValueError(
                    "Fixing frame sampling is for test time only. can remove but..."
                )
            self._fix_temporal_samples()

        self.latent_dirname = latent_dirname

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError(
            "Get video path function must be implemented by subclass"
        )

    def _get_caption_cond(self, sample):
        raise NotImplementedError(
            "Get caption function must be implemented by subclass"
        )

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata["vlen"] = self._get_video_lens()
        self.metadata["frame_intervals"] = self.metadata["vlen"].apply(
            lambda x: np.linspace(
                start=0, stop=x, num=min(x, self.video_params["num_frames"]) + 1
            ).astype(int)
        )
        self.metadata["fix_start"] = self.metadata["frame_intervals"].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode("fix_start")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):

        imgs = None
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption, cond_image_path, segmentation, bbox = self._get_caption_cond(sample)
        if isinstance(caption, float) or isinstance(caption, int):
            caption = str(caption)
        if caption is None or len(caption) == 0 or caption == '':
            print(f"Warning: missing caption for {video_fp}.")
            return self.__getitem__(np.random.choice(self.__len__()))

        video_loading = self.video_params.get("loading", "strict")
        fix_start = None if self.split == 'train' else 0
        if self.sliding_window_stride != -1:
            fix_start = sample["fix_start"]

        # Load from pre-computed latent if available
        try:
            if self.latent_dirname is not None:
                rel_fp_latent = os.path.join(self.latent_dirname, rel_fp.split('/')[-1])
                latent_fp = os.path.join(self.data_dir, rel_fp_latent).replace('.mp4', '.safetensors')
                if os.path.exists(latent_fp):
                    tensors = safetensors.torch.load_file(latent_fp)
                    latent = tensors['z'].squeeze(0)
                    if tensors['n_frames'].item() < self.video_params["num_frames"]:
                        print(f"Warning: latent file {latent_fp} has less frames than required. Sampling another video.")
                        return self.__getitem__(np.random.choice(self.__len__()))
                    if tensors['n_frames'].item() % latent.shape[1] != 0:
                        print(f"Warning: latent file {latent_fp} has inconsistent number of frames. Sampling another video.")
                        return self.__getitem__(np.random.choice(self.__len__()))
                    tdf = tensors['n_frames'].item() // latent.shape[1]
                    nframes = self.video_params["num_frames"] // tdf
                    start = np.random.randint(0, latent.shape[1] + 1 - nframes)
                    latent = latent[:, start:start+nframes]
                    if self.use_t5_latent:
                        if self.t5_latent_dir is not None:
                            t5_latent_path = os.path.join(self.t5_latent_dir, rel_fp.split('/')[-1].replace('mp4', 'pt'))
                            if not os.path.exists(t5_latent_path):
                                print(f"Warning: missing T5 latent file {t5_latent_path}. load in old")
                                t5_latent_path = t5_latent_path.replace('T5_latent', 'T5_latent_old')
                            try:
                                t5_latent = torch.load(t5_latent_path)
                            except Exception as e:
                                print(f"Warning: failed to load T5 latent file {t5_latent_path}.")
                                get_parent_dir = lambda path: os.path.join(os.path.dirname(os.path.abspath(path)), '')
                                t5_latent_path = os.path.join(get_parent_dir(self.t5_latent_dir), 't5_empty_77tokens.pt')
                                t5_latent = torch.load(t5_latent_path)
                            if isinstance(t5_latent, dict):
                                t5_latent = t5_latent['y']
                            assert isinstance(t5_latent, torch.Tensor)
                            while len(t5_latent.shape) > 2:
                                t5_latent = t5_latent.squeeze(0)
                    data = {
                        self.first_stage_key: latent,
                        self.cond_stage_key: caption,
                        "meta": {
                            "raw_captions": caption,
                            "paths": latent_fp,
                            "dataset": self.dataset_name,
                        },
                    }
                    if self.use_t5_latent:
                        data['t5_latent'] = t5_latent
                    return data
                else:
                    print(f"Warning: missing latent file {latent_fp}.")
                    return self.__getitem__(np.random.choice(self.__len__()))
        except Exception as e:
            if video_loading == "strict":
                raise ValueError(
                    f"Video loading failed for {video_fp}, video loading for this dataset is strict."
                ) from e
            else:
                print(f"Warning: unknown error in loading latent file {latent_fp}. Resampling another video.")
                return self.__getitem__(np.random.choice(self.__len__()))

        try:
            if os.path.isfile(video_fp):
                if self.frame_sample == "equally spaced":
                    sample_factor = self.video_params.get("es_interval", 10)
                elif self.frame_sample in ("proportional", "middle", "middle_random", "middle_extend_random"):
                    sample_factor = self.video_params.get("prop_factor", 3)
                imgs, idxs = self.video_reader(
                    video_fp,
                    self.video_params["num_frames"],
                    self.frame_sample,
                    fix_start=fix_start,
                    sample_factor=sample_factor,
                    max_longedge_of_load=self.video_reader_max_longedge_of_load,
                    min_shortedge_of_load=self.video_reader_min_shortedge_of_load,
                )
            else:
                print_str = f"Warning: missing video file {video_fp}."
                if video_fp not in self.lack_files:
                    self.lack_files.append(video_fp)
                if self.skip:
                    print_str += " Resampling another video."
                    print(print_str)
                    return self.__getitem__(np.random.choice(self.__len__()))
                else:
                    print(print_str)
                    assert False

        except Exception as e:
            if video_loading == "strict":
                raise ValueError(
                    f"Video loading failed for {video_fp}, video loading for this dataset is strict."
                ) from e
            else:
                traceback.print_exc()
                print("Warning: using the pure black image as the frame sample")
                imgs = Image.new(
                    "RGB",
                    (
                        self.video_params["input_res_w"],
                        self.video_params["input_res_h"],
                    ),
                    (0, 0, 0),
                )
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        meta_arr = {
            "raw_captions": caption,
            "paths": rel_fp,
            "dataset": self.dataset_name,
        }

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros(
            [
                self.video_params["num_frames"],
                3,
                self.video_params["input_res_h"],
                self.video_params["input_res_w"],
            ]
        )
        final[: imgs.shape[0]] = imgs
        final = final.permute(1, 0, 2, 3)  # CTHW

        if self.use_t5_latent:
            if self.t5_latent_dir is not None:
                t5_latent_path = os.path.join(self.t5_latent_dir, rel_fp.split('/')[-1].replace('mp4', 'pt'))
                if not os.path.exists(t5_latent_path):
                    print(f"Warning: missing T5 latent file {t5_latent_path}. load in old")
                    t5_latent_path = t5_latent_path.replace('T5_latent', 'T5_latent_old')
                try:
                    t5_latent = torch.load(t5_latent_path)
                except Exception as e:
                    print(f"Warning: failed to load T5 latent file {t5_latent_path}.")
                    get_parent_dir = lambda path: os.path.join(os.path.dirname(os.path.abspath(path)), '')
                    t5_latent_path = os.path.join(get_parent_dir(self.t5_latent_dir), 't5_empty_77tokens.pt')
                    t5_latent = torch.load(t5_latent_path)
                if isinstance(t5_latent, dict):
                    t5_latent = t5_latent['y']
                assert isinstance(t5_latent, torch.Tensor)
                while len(t5_latent.shape) > 2:
                    t5_latent = t5_latent.squeeze(0)

        # Load condition image
        try:
            cond_image = Image.open(cond_image_path).convert('RGB')
            if self.drop_mask is not None and random.random() < self.drop_mask:
                cond_image = Image.open(cond_image_path).convert('RGB')
            else:
                if self.data_augmentation:
                    cond_image = augment_object(cond_image, segmentation, bbox)
                else:
                    binary_mask = mask_util.decode(segmentation)
                    binary_mask = binary_mask.astype(bool)
                    cond_image_np = np.array(cond_image)
                    cond_image_np = np.where(binary_mask[:, :, None], cond_image_np, np.array([127, 127, 127], dtype=cond_image_np.dtype))
                    cond_image = Image.fromarray(cond_image_np.astype(np.uint8))

            cond_image = self.transforms_cond(cond_image)  # (C, H, W)
            if self.data_augmentation:
                if random.random() < 0.2:
                    cond_image = cond_image.unsqueeze(0).unsqueeze(0)
                    image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=cond_image.device)
                    image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=cond_image.dtype)
                    noisy_images = cond_image + torch.randn_like(cond_image) * image_noise_sigma[:, None, None, None, None]
                    noisy_images = noisy_images.squeeze(0).squeeze(0)
                else:
                    noisy_images = cond_image
                cond_image = noisy_images
        except Exception as e:
            print(f"Warning: failed to load condition image {cond_image_path}. Using pure grey image instead.")
            traceback.print_exc()
            cond_image = Image.new(
                    "RGB",
                    (
                        self.video_params["input_res_w"],
                        self.video_params["input_res_h"],
                    ),
                    (127, 127, 127),
                )
            cond_image = transforms.ToTensor()(cond_image)

        data = {
            self.first_stage_key: final,
            self.cond_stage_key: caption,
            "meta": meta_arr,
        }
        data['cond_image'] = cond_image
        if self.use_t5_latent:
            data['t5_latent'] = t5_latent

        return data


class WebVidPT2V(TextVideoDatasetPT2V):
    """
    PexelsCustom-1M Dataset.
    Loads videos with per-object condition images and segmentation annotations.
    """

    def _load_metadata(self):
        if self.metadata_path is None:
            assert self.metadata_folder_name is not None
            assert self.cut is not None
            metadata_dir = os.path.join(
                self.metadata_dir, self.metadata_folder_name)
            if self.key is None:
                metadata_fp = os.path.join(
                    metadata_dir, f"results_{self.cut}_{self.split}.csv"
                )
            else:
                metadata_fp = os.path.join(
                    metadata_dir, f"results_{self.cut}_{self.split}_{self.key}.csv"
                )
        else:
            metadata_fp = self.metadata_path
        print(metadata_fp)
        metadata = pd.read_csv(
            metadata_fp,
            on_bad_lines="skip",
            encoding="ISO-8859-1",
            engine="python",
            sep=",",
        )

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == "val":
            try:
                metadata = metadata.sample(1000, random_state=0)
                print("Downsampled val set to 1000 samples")
            except:
                print(
                    "there are less than 1000 samples in the val set, thus no downsampling is done"
                )
                pass
        if "name" in metadata.columns:
            metadata["caption"] = metadata["name"]
            del metadata["name"]
        self.metadata = metadata
        print(f"Dataload: Loaded {len(self.metadata)} samples from {metadata_fp}")

    def _get_video_path(self, sample):
        rel_video_fp = str(sample["videoid"]) + ".mp4" if len(str(sample["videoid"]).split('.')) == 1 else str(sample["videoid"])
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption_cond(self, sample):
        videoid = sample['videoid']
        object_index = int(sample['object_index'])
        filename = os.path.basename(videoid)
        dir_name = os.path.dirname(videoid)
        cond_image_path = os.path.join(self.cond_image_dir, dir_name, filename, 'original.png')
        if self.is_filter:
            root_dir = sample['root_dir']
            json_path = os.path.join(root_dir, dir_name, filename, str(filename) + '.json')
        else:
            json_path = os.path.join(self.cond_json_dir, dir_name, filename, str(filename) + '.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        annotation = data['annotations'][object_index]
        bbox = annotation['bbox']
        if not self.use_recaption:
            caption = str(data['original_caption'])
        else:
            if self.is_filter:
                caption = str(sample['personalized_caption'])
            else:
                caption = str(annotation['caption'])
        segmentation = annotation['segmentation']
        return caption, cond_image_path, segmentation, bbox


def build_dataloader(
    size=(320, 512),
    batch_size=4,
    num_workers=4,
    data_dir='',
    metadata_dir='',
    split='train',
    shuffle=False,
    num_frames=16,
    fps=24,
    subsample=1,
    tsfm_type="standard",
    max_longedge_of_load=None,
    return_dataset=False,
    latent_dirname=None,
    t5_latent_dir=None,
    use_t5_latent=False,
    frame_sample="middle",
    metadata_path=None,
    cond_image_dir=None,
    cond_json_dir=None,
    data_augmentation=False,
    drop_mask=None,
    use_recaption=True,
    is_filter=False,
    **kwargs,
):
    if split in ('train', 'pexels-train'):
        split = 'train'
        cut = None
        key = None
        metadata_folder_name = None
    elif split in ('val', 'pexels-val'):
        split = 'val'
        cut = None
        key = None
        metadata_folder_name = None
    else:
        raise ValueError(f"split {split} not recognized")

    h, w = size
    config = {
        "dataset_name": "PexelsCustom",
        "data_dir": data_dir,
        "metadata_dir": metadata_dir,
        "split": split,
        "cut": cut,
        "key": key,
        "subsample": subsample,
        "text_params": {"input": "text"},
        "video_params": {
            "input_res_h": h,
            "input_res_w": w,
            "tsfm_type": tsfm_type,
            "tsfm_params": {
                "norm_mean": [0.5, 0.5, 0.5],
                "norm_std": [0.5, 0.5, 0.5],
            },
            "num_frames": num_frames,
            "prop_factor": fps,
            "loading": "lax",
            "max_longedge_of_load": max_longedge_of_load,
            "frame_sample": frame_sample
        },
        "metadata_folder_name": metadata_folder_name,
        "first_stage_key": "image",
        "cond_stage_key": "caption",
        "skip_missing_files": True,
        "latent_dirname": latent_dirname,
        "t5_latent_dir": t5_latent_dir,
        "use_t5_latent": use_t5_latent,
        "cond_image_dir": cond_image_dir,
        "cond_json_dir": cond_json_dir,
        "metadata_path": metadata_path,
        "data_augmentation": data_augmentation,
        "drop_mask": drop_mask,
        "use_recaption": use_recaption,
        "is_filter": is_filter,
    }

    dataset = WebVidPT2V(**config)
    if return_dataset:
        return dataset

    dataloader = prepare_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        pg_manager=get_parallel_manager(),
    )
    return dataloader


def prod(nums: List[int]) -> int:
    """Product of a list of numbers."""
    return reduce(mul, nums)


class ProcessGroupMesh:
    """A helper class to manage the process group mesh.

    We use a ND-tuple to represent the process group mesh. And a ND-coordinate is to represent each process.

    Args:
        *size (int): The size of each dimension of the process group mesh.
    """

    def __init__(self, *size: int) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."
        assert prod(size) == dist.get_world_size(), "The product of the size must be equal to the world size."
        self._shape = size
        self._rank = dist.get_rank()
        self._coord = ProcessGroupMesh.unravel(self._rank, self._shape)
        self._ranks_to_group: Dict[Tuple[int, ...], ProcessGroup] = {}
        self._group_to_ranks: Dict[ProcessGroup, Tuple[int, ...]] = {}

    def destroy_mesh_process_groups(self):
        for group in self._ranks_to_group.values():
            dist.destroy_process_group(group)
        gc.collect()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def rank(self) -> int:
        return self._rank

    def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        if dim is None:
            return self._shape
        else:
            return self._shape[dim]

    def coordinate(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        if dim is None:
            return self._coord
        else:
            return self._coord[dim]

    @staticmethod
    def unravel(rank: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return np.unravel_index(rank, shape)

    @staticmethod
    def ravel(coord: Tuple[int, ...], shape: Tuple[int, ...], mode: str = "raise") -> int:
        assert mode in ["raise", "wrap", "clip"]
        return np.ravel_multi_index(coord, shape, mode)

    def get_group(self, ranks_in_group: List[int], backend: Optional[str] = None) -> ProcessGroup:
        ranks_in_group = sorted(ranks_in_group)
        if tuple(ranks_in_group) not in self._group_to_ranks:
            group = dist.new_group(ranks_in_group, backend=backend)
            self._ranks_to_group[tuple(ranks_in_group)] = group
            self._group_to_ranks[group] = tuple(ranks_in_group)
        return self._ranks_to_group[tuple(ranks_in_group)]

    def get_ranks_in_group(self, group: ProcessGroup) -> List[int]:
        return list(self._group_to_ranks[group])

    @staticmethod
    def get_coords_along_axis(
        base_coord: Tuple[int, ...], axis: int, indices_at_axis: List[int]
    ) -> List[Tuple[int, ...]]:
        coords_in_group = []
        for idx in indices_at_axis:
            coords_in_group.append(base_coord[:axis] + (idx,) + base_coord[axis + 1 :])
        return coords_in_group

    def create_group_along_axis(
        self, axis: int, indices_at_axis: Optional[List[int]] = None, backend: Optional[str] = None
    ) -> ProcessGroup:
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        reduced_shape = list(self._shape)
        reduced_shape[axis] = 1
        target_group = None
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords_in_group = ProcessGroupMesh.get_coords_along_axis(base_coord, axis, indices_at_axis)
            ranks_in_group = tuple([ProcessGroupMesh.ravel(coord, self._shape) for coord in coords_in_group])
            group = self.get_group(ranks_in_group, backend=backend)
            if self._rank in ranks_in_group:
                target_group = group
        return target_group

    def get_group_along_axis(
        self, axis: int, indices_at_axis: Optional[List[int]] = None, backend: Optional[str] = None
    ) -> ProcessGroup:
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        coords_in_group = ProcessGroupMesh.get_coords_along_axis(self._coord, axis, indices_at_axis)
        ranks_in_group = tuple([ProcessGroupMesh.ravel(coord, self._shape) for coord in coords_in_group])
        if ranks_in_group not in self._ranks_to_group:
            return self.create_group_along_axis(axis, indices_at_axis, backend=backend)
        return self._ranks_to_group[ranks_in_group]


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


def prepare_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    pg_manager=None,
    collate_fn=None,
    **kwargs,
):
    """
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.

    Args:
        dataset: The dataset to be loaded.
        shuffle: Whether to shuffle the dataset. Defaults to False.
        seed: Random worker seed for sampling, defaults to 1024.
        drop_last: Set to True to drop the last incomplete batch. Defaults to False.
        pin_memory: Whether to pin memory address in CPU memory. Defaults to False.
        num_workers: Number of worker threads for this dataloader. Defaults to 0.

    Returns:
        A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
        **_kwargs,
    )


PARALLEL_MANAGER = None


class ParallelManager(ProcessGroupMesh):
    def __init__(self, dp_size, sp_size, dp_axis, sp_axis):
        super().__init__(dp_size, sp_size)
        self.dp_axis = dp_axis
        self.dp_group: ProcessGroup = self.get_group_along_axis(self.dp_axis)
        self.dp_rank = dist.get_rank(self.dp_group)

        self.sp_size = sp_size
        self.sp_axis = sp_axis
        self.sp_group: ProcessGroup = self.get_group_along_axis(self.sp_axis)
        self.sp_rank = dist.get_rank(self.sp_group)
        self.enable_sp = sp_size > 1


def set_parallel_manager(dp_size, sp_size, dp_axis, sp_axis):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(dp_size, sp_size, dp_axis, sp_axis)


def get_data_parallel_group():
    return PARALLEL_MANAGER.dp_group


def get_data_parallel_rank():
    return PARALLEL_MANAGER.dp_rank


def get_sequence_parallel_group():
    return PARALLEL_MANAGER.sp_group


def get_sequence_parallel_size():
    return PARALLEL_MANAGER.sp_size


def get_sequence_parallel_rank():
    return PARALLEL_MANAGER.sp_rank


def use_sequence_parallelism():
    return PARALLEL_MANAGER.enable_sp


def get_parallel_manager():
    return PARALLEL_MANAGER
