import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora_controller import enable_lora


class PT2VLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor, cond_hidden_states: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with enable_lora([self.linear], False):
            shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        if cond_hidden_states is not None:
            cond_shift, cond_scale, cond_gate, _, _, _ = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        if cond_hidden_states is not None:
            cond_hidden_states = self.norm(cond_hidden_states) * (1 + cond_scale)[:, None, :] + cond_shift[:, None, :]
            return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :], cond_hidden_states, cond_gate[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]
