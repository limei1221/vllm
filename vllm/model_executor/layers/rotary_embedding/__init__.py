# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings."""

from typing import Any

import torch

from .base import RotaryEmbedding
from .deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
    DeepseekV4ScalingRotaryEmbedding,
)

_ROPE_DICT: dict[tuple[Any, ...], RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_parameters is not None:
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_parameters.items()
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())
    else:
        rope_parameters_args = None

    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000)
    scaling_type = rope_parameters.get("rope_type", "default")
    if rotary_dim := rope_parameters.get("rope_dim", None):
        pass
    else:
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
            raise ValueError(f"{partial_rotary_factor=} must be between 0.0 and 1.0")
        rotary_dim = int(head_size * partial_rotary_factor)

    key = (
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if scaling_type == "default":
        rotary_emb = RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
        )
    elif scaling_type in ["deepseek_yarn", "deepseek_llama_scaling"]:
        scaling_factor = rope_parameters["factor"]
        original_max_position = rope_parameters["original_max_position_embeddings"]
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
        }
        if rope_parameters.get("is_deepseek_v4", False):
            cls = DeepseekV4ScalingRotaryEmbedding
        else:
            cls = DeepseekScalingRotaryEmbedding
        rotary_emb = cls(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )
    else:
        raise ValueError(f"Unsupported RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
