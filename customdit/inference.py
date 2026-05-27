"""
Inference script for CustomDiT — generate identity-preserving videos.

Usage:
    python inference.py --model_path THUDM/CogVideoX-5b --lora_path path/to/lora.safetensors --csv_path samples.csv --output_dir ./output
"""

import logging
import argparse
from typing import Optional

import os
import torch
from diffusers import CogVideoXDPMScheduler
from customdit.model.pt2v_transformer_3d import PT2VTransformer3DModel
from customdit.model.pipeline_pt2v import PT2VPipeline
import pandas as pd
from safetensors.torch import load_file
from diffusers.utils import export_to_video

from omegaconf import OmegaConf
from peft import LoraConfig, set_peft_model_state_dict
from diffusers.utils import convert_unet_state_dict_to_peft

logging.basicConfig(level=logging.INFO)

# Recommended resolution for CogVideoX-5b (width, height)
RESOLUTION_MAP = {
    "cogvideox-5b": (720, 480),
}


def generate_video(
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 49,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 8,
    csv_path: str = None,
    output_dir: str = None,
    lora_config: LoraConfig = None,
    use_negative_prompt: bool = False,
):
    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP.get(model_name, (720, 480))
    if width is None or height is None:
        width, height = desired_resolution
        logging.info(f"Using default resolution {desired_resolution} for {model_name}")
    elif (width, height) != desired_resolution:
        logging.warning(f"{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.")
        width, height = desired_resolution

    if lora_path:
        transformer = PT2VTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        adapter_name = "default"
        transformer.add_adapter(adapter_name=adapter_name, adapter_config=lora_config)
        transformer.to('cuda')

        lora_state_dict_trained = load_file(lora_path)
        lora_state_dict = PT2VPipeline.lora_state_dict(lora_state_dict_trained)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        if len(transformer_state_dict) == 0:
            transformer_state_dict = lora_state_dict
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
                logging.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        transformer.requires_grad_(False)
        transformer.eval()
        pipe = PT2VPipeline.from_pretrained(model_path, torch_dtype=dtype, transformer=transformer)
        logging.info(f"LoRA loaded from {lora_path}")
    else:
        pipe = PT2VPipeline.from_pretrained(model_path, torch_dtype=dtype)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")

    df = pd.read_csv(csv_path)
    if use_negative_prompt:
        negative_prompt = "blurry, low quality, overexposed, underexposed, blurry background, messy background, unnatural background, grain, noise, artifacts, extra limbs, deformed, out of frame, distorted, poorly drawn, over-edited, motion incoherent, lecture style, watermark, screen noise"
    else:
        negative_prompt = None

    for i, row in df.iterrows():
        caption = row['caption']
        cond_image_path = row['cond_image_path']
        video_generate = pipe(
            height=height,
            width=width,
            prompt=caption,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            cond_image_path=cond_image_path,
        ).frames[0]
        caption_short = caption[:200] if len(caption) > 200 else caption
        output_path = os.path.join(output_dir, f"{os.path.basename(cond_image_path).split('.')[0]}_{caption_short}.mp4")
        export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate identity-preserving videos with CustomDiT")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5b", help="Path of the pre-trained model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights (.safetensors)")
    parser.add_argument("--lora_rank", type=int, default=128, help="Rank of the LoRA weights")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to generate")
    parser.add_argument("--width", type=int, default=None, help="Width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="Height of the generated video")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV with columns: caption, cond_image_path")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for generated videos")
    parser.add_argument("--lora_config", type=str, default=None, help="Path to YAML config with lora_config section")
    parser.add_argument("--use_negative_prompt", action="store_true", help="Use a default negative prompt")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)

    transformer_lora_config = None
    if args.lora_config:
        config = OmegaConf.load(args.lora_config)
        transformer_lora_config = LoraConfig(**config.lora_config)

    generate_video(
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        lora_config=transformer_lora_config,
        use_negative_prompt=args.use_negative_prompt,
    )
