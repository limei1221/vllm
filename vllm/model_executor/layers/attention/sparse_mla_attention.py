# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared MHA implementation and metadata builder for sparse MLA backends."""

from typing import TYPE_CHECKING, TypeVar

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)

GLOBAL_TOPK_MASK_MAX_BYTES = 128 * 1024 * 1024  # 128 MiB


@triton.jit
def _scatter_topk_kernel(
    mask_ptr,
    topk_ptr,
    cu_q_lens_ptr,
    num_words: tl.constexpr,
    mask_row_stride: tl.constexpr,
    num_topk: tl.constexpr,
    topk_stride: tl.constexpr,
    max_q_len: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = mask_ptr + row_idx * mask_row_stride
    word_offsets = tl.arange(0, BLOCK_WORDS)
    tl.store(
        row_ptr + word_offsets,
        tl.zeros([BLOCK_WORDS], dtype=tl.int32),
        mask=word_offsets < num_words,
    )

    req_idx = row_idx // max_q_len
    q_local = row_idx % max_q_len
    q_start = tl.load(cu_q_lens_ptr + req_idx)
    q_len = tl.load(cu_q_lens_ptr + req_idx + 1) - q_start
    if q_local < q_len:
        src_row = q_start + q_local
        offsets = tl.arange(0, BLOCK_TOPK)
        in_range = offsets < num_topk
        indices = tl.load(
            topk_ptr + src_row * topk_stride + offsets,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (indices >= 0)
        word_indices = indices >> 5
        bits = (1 << (indices & 31)).to(tl.int32)
        tl.atomic_or(row_ptr + word_indices, bits, mask=valid)


@triton.jit
def _scatter_topk_single_req_kernel(
    mask_ptr,
    topk_ptr,
    num_words: tl.constexpr,
    mask_row_stride: tl.constexpr,
    num_topk: tl.constexpr,
    topk_stride: tl.constexpr,
    total_q: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = mask_ptr + row_idx * mask_row_stride
    word_offsets = tl.arange(0, BLOCK_WORDS)
    tl.store(
        row_ptr + word_offsets,
        tl.zeros([BLOCK_WORDS], dtype=tl.int32),
        mask=word_offsets < num_words,
    )
    if row_idx < total_q:
        offsets = tl.arange(0, BLOCK_TOPK)
        in_range = offsets < num_topk
        indices = tl.load(
            topk_ptr + row_idx * topk_stride + offsets,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (indices >= 0)
        word_indices = indices >> 5
        bits = (1 << (indices & 31)).to(tl.int32)
        tl.atomic_or(row_ptr + word_indices, bits, mask=valid)
