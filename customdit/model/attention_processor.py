import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.attention_processor import Attention
from .lora_controller import enable_lora


class PT2VAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the PT2V model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("PT2VAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        video_seq_length = None,
        cond_image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        hidden_states, cond_hidden_states = hidden_states.split([text_seq_length + video_seq_length, hidden_states.size(1) - text_seq_length - video_seq_length], dim=1)

        with enable_lora([attn.to_q, attn.to_k, attn.to_v], False):
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        cond_query = attn.to_q(cond_hidden_states)
        cond_key = attn.to_k(cond_hidden_states)
        cond_value = attn.to_v(cond_hidden_states)

        query = torch.cat([query, cond_query], dim=1)
        key = torch.cat([key, cond_key], dim=1)
        value = torch.cat([value, cond_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb
            query[:, :, text_seq_length:text_seq_length+video_seq_length] = apply_rotary_emb(query[:, :, text_seq_length:text_seq_length+video_seq_length], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:text_seq_length+video_seq_length] = apply_rotary_emb(key[:, :, text_seq_length:text_seq_length+video_seq_length], image_rotary_emb)
            if cond_image_rotary_emb is not None:
                query[:, :, text_seq_length+video_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length+video_seq_length:], cond_image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, text_seq_length+video_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length+video_seq_length:], cond_image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states, cond_hidden_states = hidden_states.split([text_seq_length + video_seq_length, hidden_states.size(1) - text_seq_length - video_seq_length], dim=1)
        # linear proj
        with enable_lora([attn.to_out[0]], False):
            hidden_states = attn.to_out[0](hidden_states)
        cond_hidden_states = attn.to_out[0](cond_hidden_states)
        hidden_states = torch.cat([hidden_states, cond_hidden_states], dim=1)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
