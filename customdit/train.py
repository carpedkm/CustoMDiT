"""
Training script for CustomDiT (LoRA fine-tuning on CogVideoX-5b).

Usage:
    accelerate launch --config_file configs/accelerate_single_node.yaml train.py --config configs/train.yaml
"""

# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from omegaconf import OmegaConf
import random
from safetensors.torch import load_file
import re
import contextlib

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
import time

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from customdit.model.pt2v_transformer_3d import PT2VTransformer3DModel
from customdit.model.pipeline_pt2v import PT2VPipeline
from customdit.util.dataset import build_dataloader
from customdit.util.util import get_sub_models, get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from accelerate.state import AcceleratorState
from transformers.utils import ContextManagers
import accelerate
from transformers.integrations import is_deepspeed_zero3_enabled
import deepspeed


def get_deepspeed_plugin():
    if accelerate.state.is_initialized():
        return AcceleratorState().deepspeed_plugin
    else:
        return None


def deepspeed_zero_init_disabled_context_manager():
    """
    Returns either a context list that includes one that will disable zero.Init or an empty context list.
    """
    deepspeed_plugin = get_deepspeed_plugin()
    if deepspeed_plugin is None:
        return []
    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


if is_wandb_available():
    import wandb

check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def get_args_from_config():
    parser = argparse.ArgumentParser(description="Training script for CustomDiT.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Path to the config file containing all the training parameters.",
    )
    args, unknown = parser.parse_known_args()
    config = OmegaConf.load(args.config)
    return config


def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.validation.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.training.seed) if args.training.seed else None

    videos = []
    for _ in range(args.validation.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.training.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    free_memory()
    return videos


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer, text_encoder, prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device, dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer, text_encoder, prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device, dtype=dtype,
            )
    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = 1,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
    cond_image_pe=False,
):
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    p_t = patch_size_t
    base_num_frames = (num_frames + p_t - 1) // p_t

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
    )
    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)

    if cond_image_pe:
        start, stop = grid_crops_coords
        start, stop = list(start), list(stop)
        start[1] += grid_width
        stop[1] += grid_width
        start, stop = tuple(start), tuple(stop)
        grid_crops_coords_cond_image = (start, stop)
        freqs_cos_cond, freqs_sin_cond = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords_cond_image,
            grid_size=(grid_height, grid_width),
            temporal_size=1,
        )
        return freqs_cos, freqs_sin, freqs_cos_cond, freqs_sin_cond

    return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    if use_deepspeed:
        from accelerate.utils import DummyOptim
        return DummyOptim(
            params_to_optimize,
            lr=args.training.learning_rate,
            betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
            eps=args.optimizer.adam_epsilon,
            weight_decay=args.optimizer.adam_weight_decay,
        )

    supported_optimizers = ["adam", "adamw", "prodigy"]
    args.optimizer.type = args.optimizer.type.lower()
    if args.optimizer.type not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer.type = "adamw"

    if args.optimizer.use_8bit_adam and not (args.optimizer.type.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.type.lower()}"
        )

    if args.optimizer.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.type.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.optimizer.use_8bit_adam else torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.training.learning_rate,
            betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
            eps=args.optimizer.adam_epsilon,
            weight_decay=args.optimizer.adam_weight_decay,
        )
    elif args.optimizer.type.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.optimizer.use_8bit_adam else torch.optim.Adam
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.training.learning_rate,
            betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
            eps=args.optimizer.adam_epsilon,
            weight_decay=args.optimizer.adam_weight_decay,
        )
    elif args.optimizer.type.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.training.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.training.learning_rate,
            betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
            beta3=args.optimizer.prodigy_beta3,
            weight_decay=args.optimizer.adam_weight_decay,
            eps=args.optimizer.adam_epsilon,
            decouple=args.optimizer.prodigy_decouple,
            use_bias_correction=args.optimizer.prodigy_use_bias_correction,
            safeguard_warmup=args.optimizer.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if torch.backends.mps.is_available() and args.training.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.training.output_dir, args.logging.logging_dir)

    expected_mixed_precision = "bf16" if "5b" in args.pretrained_model.name_or_path.lower() else "fp16"
    if args.training.mixed_precision != expected_mixed_precision:
        raise ValueError(f"Mixed precision {args.training.mixed_precision} does not match the model precision, should be {expected_mixed_precision}")

    accelerator_project_config = ProjectConfiguration(project_dir=args.training.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    if hasattr(args.training, "debug") and args.training.debug:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.training.gradient_accumulation_steps,
            mixed_precision=args.training.mixed_precision,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.training.gradient_accumulation_steps,
            mixed_precision=args.training.mixed_precision,
            log_with=args.logging.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

    if accelerator.state.deepspeed_plugin:
        config = {
            'optimizer': {
                'type': args.optimizer.type,
                'params': {
                    'lr': args.training.learning_rate,
                    'betas': [args.optimizer.adam_beta1, args.optimizer.adam_beta2]
                },
                'torch_adam': True
            },
            'bf16': {
                'enabled': True if args.training.mixed_precision == "bf16" else False
            },
            'fp16': {
                'enabled': True if args.training.mixed_precision == "fp16" else False
            },
            'gradient_accumulation_steps': args.training.gradient_accumulation_steps,
        }
        accelerator.state.deepspeed_plugin.deepspeed_config.update(config)

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.logging.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.training.seed is not None:
        set_seed(args.training.seed)

    if args.training.output_dir is not None:
        os.makedirs(args.training.output_dir, exist_ok=True)

    # CogVideoX-5b weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model.name_or_path.lower() else torch.float16

    transformer = PT2VTransformer3DModel.from_pretrained(
        args.pretrained_model.name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.pretrained_model.revision,
        variant=args.pretrained_model.variant,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model.name_or_path, subfolder="scheduler")

    transformer.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    transformer.to(accelerator.device, dtype=weight_dtype)

    if args.training.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Add LoRA adapter
    transformer_lora_config = LoraConfig(**args.lora_config)
    adapter_name = "default"
    transformer.add_adapter(adapter_name=adapter_name, adapter_config=transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if is_deepspeed_zero3_enabled():
            for model in models:
                model = unwrap_model(model)
                params_to_gather = [(name, param) for name, param in model.named_parameters() if 'lora' in name]
                transformer_lora_layers_to_save = {}
                for name, param in params_to_gather:
                    context_manager = deepspeed.zero.GatheredParameters(param, modifier_rank=None)
                    with context_manager:
                        param = param.to('cpu')
                        transformer_lora_layers_to_save[name] = param
                if accelerator.is_main_process:
                    PT2VPipeline.save_lora_weights(
                        output_dir,
                        transformer_lora_layers=transformer_lora_layers_to_save,
                    )
        else:
            if accelerator.is_main_process:
                transformer_lora_layers_to_save = None
                for model in models:
                    model = unwrap_model(model)
                    if isinstance(model, type(unwrap_model(transformer))):
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
                    if len(weights) != 0:
                        weights.pop()

                PT2VPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.training.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.training.scale_lr:
        args.training.learning_rate = (
            args.training.learning_rate * args.training.gradient_accumulation_steps * args.training.train_batch_size * accelerator.num_processes
        )

    if args.training.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # Load LoRA weights from resume_path if specified
    if hasattr(args.pretrained_model, 'resume_path') and args.pretrained_model.resume_path:
        lora_path = args.pretrained_model.resume_path
        lora_state_dict_trained = load_file(lora_path)
        lora_state_dict = PT2VPipeline.lora_state_dict(lora_state_dict_trained)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        if "module." in list(transformer_state_dict.keys())[0]:
            transformer_state_dict = {
                f'{k[7:]}': v for k, v in transformer_state_dict.items() if k.startswith("module.")
            }
        transformer_state_dict = {
            f'{k.replace(".default.weight", ".weight")}': v for k, v in transformer_state_dict.items()
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name=adapter_name)
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        logger.info(f"Loaded LoRA weights from {lora_path}")
    else:
        if args.training.resume_from_checkpoint:
            if args.training.resume_from_checkpoint != "latest":
                path = os.path.basename(args.training.resume_from_checkpoint)
            else:
                try:
                    dirs = os.listdir(args.training.output_dir)
                except FileNotFoundError:
                    os.makedirs(args.training.output_dir, exist_ok=True)
                    dirs = os.listdir(args.training.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is not None:
                logger.info(f"Loading LoRA from checkpoint {path}")
                lora_path = os.path.join(args.training.output_dir, path, "pytorch_lora_weights.safetensors")
                lora_state_dict_trained = load_file(lora_path)
                lora_state_dict = PT2VPipeline.lora_state_dict(lora_state_dict_trained)

                transformer_state_dict = {
                    f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
                }
                if "module." in list(transformer_state_dict.keys())[0]:
                    transformer_state_dict = {
                        f'{k[7:]}': v for k, v in transformer_state_dict.items() if k.startswith("module.")
                    }
                transformer_state_dict = {
                    f'{k.replace(".default.weight", ".weight")}': v for k, v in transformer_state_dict.items()
                }
                transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
                incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name=adapter_name)
                if incompatible_keys is not None:
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(
                            f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                            f" {unexpected_keys}. "
                        )
                global_step = int(path.split("-")[1])

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.training.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    num_params_to_optimize = sum(p.numel() for p in transformer_lora_parameters)
    logger.info(f"Number of trainable parameters: {num_params_to_optimize}")

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)
    logger.info("Optimizer loaded")

    # Dataset and DataLoader
    train_dataloader = build_dataloader(**args.data_config)
    args.data_config.split = args.data_config.split.replace('train', 'val')
    args.data_config.metadata_path = args.data_config.metadata_path.replace('train', 'val')
    val_dataloader = build_dataloader(**args.data_config)

    logger.info("Training data loaded")

    # Scheduler and math around the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training.gradient_accumulation_steps)
    if args.training.max_train_steps is None:
        args.training.max_train_steps = args.training.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        args.training.lr_scheduler = True
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            total_num_steps=args.training.max_train_steps * accelerator.num_processes,
            warmup_num_steps=args.training.lr_warmup_steps * accelerator.num_processes,
        )
    elif args.training.lr_scheduler:
        lr_scheduler = get_scheduler(
            args.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.training.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.training.max_train_steps * accelerator.num_processes,
            num_cycles=args.training.lr_num_cycles,
            power=args.training.lr_power,
        )
    else:
        lr_scheduler = None

    logger.info("LR scheduler loaded")

    # Load sub-models (VAE, tokenizer, text encoder)
    vae, tokenizer, text_encoder = get_sub_models(args)

    if args.training.enable_slicing:
        vae.enable_slicing()
    if args.training.enable_tiling:
        vae.enable_tiling()
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.training.lr_scheduler:
        transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            transformer, optimizer, train_dataloader, val_dataloader
        )

    logger.info("Accelerator prepared")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.training.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.training.max_train_steps = args.training.num_train_epochs * num_update_steps_per_epoch
    args.training.num_train_epochs = math.ceil(args.training.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_name = args.logging.tracker_name or "customdit-lora"
        run_id_path = os.path.join(args.training.output_dir, 'run_id.txt')
        if os.path.exists(run_id_path):
            with open(run_id_path, "r") as f:
                run_id = f.read().strip()
        else:
            run_id = wandb.util.generate_id()
            with open(run_id_path, "w") as f:
                f.write(run_id)
        init_kwargs = {
            "wandb": {
                "resume": "allow",
                "id": run_id,
            }
        }
        accelerator.init_trackers(tracker_name, config={"test": None}, init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.training.train_batch_size * accelerator.num_processes * args.training.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataloader) * args.training.train_batch_size}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.training.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.training.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.training.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.training.resume_from_checkpoint != "latest":
            path = os.path.basename(args.training.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.training.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.training.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.training.resume_from_checkpoint = None
            initial_global_step = 0
            skipped_dataloader = train_dataloader
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.training.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            skipped_dataloader = accelerator.skip_first_batches(train_dataloader, int(path.split("-")[1]) * args.training.gradient_accumulation_steps)

    progress_bar = tqdm(
        range(0, args.training.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.training.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        for step, batch in enumerate(skipped_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                videos = batch["image"].to(accelerator.device, dtype=vae.dtype)
                model_input = vae.encode(videos).latent_dist.sample()
                model_input = model_input.permute(0, 2, 1, 3, 4).to(dtype=weight_dtype)
                model_input = model_input.to(accelerator.device)

                prompts = batch["caption"]
                if args.training.enable_text_cfg:
                    drop_rate = args.training.text_cfg_drop_rate
                    if random.random() < drop_rate:
                        prompts = [''] * len(prompts)

                cond_images = batch["cond_image"].to(dtype=weight_dtype) if "cond_image" in batch else None
                if cond_images is not None:
                    if len(cond_images.shape) == 4:
                        cond_images = cond_images.unsqueeze(2)
                    cond_images = cond_images.to(vae.device)
                    with torch.no_grad():
                        cond_images = vae.encode(cond_images).latent_dist.sample()
                        cond_images = cond_images.permute(0, 2, 1, 3, 4)
                    cond_images = cond_images.to(accelerator.device)

                prompt_embeds = compute_prompt_embeddings(
                    tokenizer, text_encoder, prompts,
                    model_config.max_text_seq_length,
                    accelerator.device, weight_dtype,
                    requires_grad=False,
                )

                noise = torch.randn_like(model_input)
                batch_size, num_frames, num_channels, height, width = model_input.shape

                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                )
                timesteps = timesteps.long()

                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.training.height,
                        width=args.training.width,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t if model_config.patch_size_t is not None else 1,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                        cond_image_pe=cond_images is not None,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )
                if cond_images is not None:
                    freqs_cos, freqs_sin, freqs_cos_cond, freqs_sin_cond = image_rotary_emb
                    image_rotary_emb = (freqs_cos, freqs_sin)
                    cond_image_rotary_emb = (freqs_cos_cond, freqs_sin_cond)
                else:
                    cond_image_rotary_emb = None

                noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                with torch.amp.autocast('cuda', dtype=weight_dtype):
                    model_output = transformer(
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        cond_image_latents=cond_images,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        cond_image_rotary_emb=cond_image_rotary_emb,
                        return_dict=False,
                    )[0]

                model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input

                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                avg_loss = accelerator.gather(loss.repeat(args.training.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.training.gradient_accumulation_steps
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.optimizer.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()
                if args.training.lr_scheduler:
                    lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if args.training.lr_scheduler:
                    logs = {"loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
                else:
                    logs = {"loss": train_loss, "lr": optimizer.param_groups[0]['lr']}
                train_loss = 0.0
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                accelerator.wait_for_everyone()

                if global_step % args.training.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.training.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.training.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.training.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.training.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.training.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.training.output_dir, f"checkpoint-{global_step}")
                    logger.info(f"Saving state to {save_path}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    # Validation
                    logger.info("Begin validation")
                    accelerator.wait_for_everyone()
                    transformer.eval()
                    for param in transformer_lora_parameters:
                        param.requires_grad = False
                    with torch.no_grad():
                        avg_loss_val = 0
                        logger.info(f"Validation dataloader length: {len(val_dataloader)}")
                        pbar_val = tqdm(range(len(val_dataloader)), desc='Validation', disable=not accelerator.is_local_main_process)
                        for step_val, batch in enumerate(val_dataloader):
                            pbar_val.update(1)
                            videos = batch["image"].to(accelerator.device, dtype=vae.dtype)
                            model_input = vae.encode(videos).latent_dist.sample()
                            model_input = model_input.permute(0, 2, 1, 3, 4).to(dtype=weight_dtype)
                            prompts = batch["caption"]
                            if args.training.enable_text_cfg:
                                drop_rate = args.training.text_cfg_drop_rate
                                if random.random() < drop_rate:
                                    prompts = [''] * len(prompts)
                            cond_images = batch["cond_image"].to(dtype=weight_dtype) if "cond_image" in batch else None
                            model_input = model_input.to(accelerator.device)
                            if cond_images is not None:
                                cond_images = cond_images.unsqueeze(2)
                                cond_images = cond_images.to(vae.device)
                                with torch.no_grad():
                                    cond_images = vae.encode(cond_images).latent_dist.sample()
                                    cond_images = cond_images.permute(0, 2, 1, 3, 4)
                                cond_images = cond_images.to(accelerator.device)

                            prompt_embeds = compute_prompt_embeddings(
                                tokenizer, text_encoder, prompts,
                                model_config.max_text_seq_length,
                                accelerator.device, weight_dtype,
                                requires_grad=False,
                            )

                            noise = torch.randn_like(model_input)
                            batch_size, num_frames, num_channels, height, width = model_input.shape

                            timesteps = torch.randint(
                                0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                            )
                            timesteps = timesteps.long()

                            image_rotary_emb = (
                                prepare_rotary_positional_embeddings(
                                    height=args.training.height,
                                    width=args.training.width,
                                    num_frames=num_frames,
                                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                                    patch_size=model_config.patch_size,
                                    patch_size_t=model_config.patch_size_t if model_config.patch_size_t is not None else 1,
                                    attention_head_dim=model_config.attention_head_dim,
                                    device=accelerator.device,
                                    cond_image_pe=cond_images is not None,
                                )
                                if model_config.use_rotary_positional_embeddings
                                else None
                            )
                            if cond_images is not None:
                                freqs_cos, freqs_sin, freqs_cos_cond, freqs_sin_cond = image_rotary_emb
                                image_rotary_emb = (freqs_cos, freqs_sin)
                                cond_image_rotary_emb = (freqs_cos_cond, freqs_sin_cond)
                            else:
                                cond_image_rotary_emb = None

                            noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                            with torch.amp.autocast('cuda', dtype=weight_dtype):
                                model_output = transformer(
                                    hidden_states=noisy_model_input,
                                    encoder_hidden_states=prompt_embeds,
                                    cond_image_latents=cond_images,
                                    timestep=timesteps,
                                    image_rotary_emb=image_rotary_emb,
                                    cond_image_rotary_emb=cond_image_rotary_emb,
                                    return_dict=False,
                                )[0]
                            model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                            alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                            weights = 1 / (1 - alphas_cumprod)
                            while len(weights.shape) < len(model_pred.shape):
                                weights = weights.unsqueeze(-1)

                            target = model_input

                            loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                            avg_loss = accelerator.gather(loss.repeat(args.training.train_batch_size)).mean()
                            avg_loss_val += avg_loss.item()

                        avg_loss_val /= step_val + 1
                        accelerator.log({'validation loss': avg_loss_val}, step=global_step)

                    transformer.train()
                    for param in transformer_lora_parameters:
                        param.requires_grad = True
                    accelerator.wait_for_everyone()

            if global_step >= args.training.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args_from_config()
    main(args)
