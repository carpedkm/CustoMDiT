"""Utility functions for CustomDiT: rotary position embeddings and model loading helpers."""

import math
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn
from PIL import Image

from transformers import AutoTokenizer, T5EncoderModel, LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers import AutoencoderKLCogVideoX, AutoencoderKLHunyuanVideo
import accelerate
from accelerate.state import AcceleratorState
from transformers.utils import ContextManagers


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    if use_real is not True:
        raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.linspace(
            start[0], stop[0] * (grid_size_h - 1) / grid_size_h, grid_size_h, device=device, dtype=torch.float32
        )
        grid_w = torch.linspace(
            start[1], stop[1] * (grid_size_w - 1) / grid_size_w, grid_size_w, device=device, dtype=torch.float32
        )
        grid_t = torch.linspace(
            0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size, device=device, dtype=torch.float32
        )
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
        grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )
        freqs_h = freqs_h[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )
        freqs_w = freqs_w[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )
        return freqs

    t_cos, t_sin = freqs_t
    h_cos, h_sin = freqs_h
    w_cos, w_sin = freqs_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin


def get_deepspeed_plugin():
    if accelerate.state.is_initialized():
        return AcceleratorState().deepspeed_plugin
    else:
        return None


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = get_deepspeed_plugin()
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def get_sub_models(args):
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model.name_or_path, subfolder="vae", revision=args.pretrained_model.revision, variant=args.pretrained_model.variant
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model.text_model_path, subfolder="tokenizer", revision=args.pretrained_model.revision
        )

        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model.text_model_path, subfolder="text_encoder", revision=args.pretrained_model.revision
        )
    return vae, tokenizer, text_encoder


def get_sub_models_hy(args):
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            args.pretrained_model.name_or_path, subfolder="vae",
        )

        text_encoder = LlamaModel.from_pretrained(
            args.pretrained_model.text_model_path, subfolder="text_encoder",
        )
        text_encoder_2 = CLIPTextModel.from_pretrained(
            args.pretrained_model.text_model_path, subfolder="text_encoder_2",
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(
            args.pretrained_model.text_model_path, subfolder="tokenizer",
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            args.pretrained_model.text_model_path, subfolder="tokenizer_2",
        )
    return vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2


def resize_and_pad(image, target_width, target_height, fill=0, interpolation=Image.BICUBIC):
    """
    Resize an image preserving aspect ratio and pad to fit (target_width, target_height).

    Args:
        image: Input PIL Image.
        target_width, target_height: Output dimensions.
        fill: Padding color value or tuple.
        interpolation: Resampling method (default: PIL.Image.BICUBIC).
    """
    w, h = image.size
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    image = image.resize((new_w, new_h), interpolation)

    pad_left = (target_width - new_w) // 2
    pad_top = (target_height - new_h) // 2
    pad_right = target_width - new_w - pad_left
    pad_bottom = target_height - new_h - pad_top

    image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill, padding_mode='constant')

    return image


def get_llama_prompt_embeds(
    prompt: Union[str, List[str]],
    prompt_template: Dict[str, Any],
    tokenizer,
    text_encoder,
    num_videos_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 256,
    num_hidden_layers_to_skip: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    prompt = [prompt_template["template"].format(p) for p in prompt]

    crop_start = prompt_template.get("crop_start", None)
    if crop_start is None:
        prompt_template_input = tokenizer(
            prompt_template["template"],
            padding="max_length",
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=False,
        )
        crop_start = prompt_template_input["input_ids"].shape[-1] - 2

    max_sequence_length += crop_start
    text_inputs = tokenizer(
        prompt,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_attention_mask = text_inputs.attention_mask.to(device)

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_attention_mask,
        output_hidden_states=True,
    ).hidden_states[-(num_hidden_layers_to_skip + 1)].to(dtype=dtype)

    if crop_start > 0:
        prompt_embeds = prompt_embeds[:, crop_start:]
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1).view(batch_size * num_videos_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(1, num_videos_per_prompt).view(batch_size * num_videos_per_prompt, seq_len)

    return prompt_embeds, prompt_attention_mask


def get_clip_prompt_embeds(
    prompt: Union[str, List[str]],
    tokenizer,
    text_encoder_2,
    num_videos_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 77,
) -> torch.Tensor:
    device = device or text_encoder_2.device
    dtype = dtype or text_encoder_2.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        print(
            f"Warning: CLIP input was truncated. Truncated parts: {removed_text}"
        )

    prompt_embeds = text_encoder_2(text_input_ids.to(device), output_hidden_states=False).pooler_output
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt).view(batch_size * num_videos_per_prompt, -1)

    return prompt_embeds


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


def encode_prompt(
    prompt: Union[str, List[str]],
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    prompt_2: Union[str, List[str]] = None,
    prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 256,
):
    if prompt_template is None:
        prompt_template = {"template": "{}"}

    if prompt_embeds is None:
        prompt_embeds, prompt_attention_mask = get_llama_prompt_embeds(
            prompt=prompt,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )

    if pooled_prompt_embeds is None:
        if prompt_2 is None:
            prompt_2 = prompt
        pooled_prompt_embeds = get_clip_prompt_embeds(
            prompt=prompt_2,
            tokenizer=tokenizer_2,
            text_encoder_2=text_encoder_2,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=77,
        )

    return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask
