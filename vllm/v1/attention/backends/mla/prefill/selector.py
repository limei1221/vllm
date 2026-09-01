# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selector for MLA prefill backends (focused Hopper build)."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

logger = init_logger(__name__)


class MLAPrefillSelectorConfig(NamedTuple):
    """Hashable configuration for MLA prefill backend selection."""

    dtype: torch.dtype
    mla_dimensions: MLADimensions = MLADimensions(
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
    )

    def __repr__(self) -> str:
        return (
            f"MLAPrefillSelectorConfig(dtype={self.dtype}, "
            f"mla_dimensions={self.mla_dimensions})"
        )


def get_mla_prefill_backend(
    vllm_config: VllmConfig,
) -> type[MLAPrefillBackend]:
    """Return the FlashAttention MLA prefill backend.

    This focused build always uses FlashAttention for prefill on Hopper.
    """
    from vllm.v1.attention.backends.mla.prefill.flash_attn import (
        FlashAttnPrefillBackend,
    )

    logger.info_once("Using FlashAttention MLA prefill backend.")
    return FlashAttnPrefillBackend


def select_mla_prefill_backend() -> type:
    """Direct selector for FlashAttention prefill backend."""
    from vllm.v1.attention.backends.mla.prefill.flash_attn import (
        FlashAttnPrefillBackend,
    )

    return FlashAttnPrefillBackend


def select_mla_decode_backend() -> type:
    """Direct selector for FlashMLA decode backend."""
    from vllm.v1.attention.backends.mla.flashmla import FlashMLABackend

    return FlashMLABackend
