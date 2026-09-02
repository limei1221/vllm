# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stride-aware FP8 quantization with head_dim padding for ViT attention.

Reads directly from non-contiguous QKV views using 3D strides and pads
head_dim to a multiple of 16 for cuDNN compatibility.
"""

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.triton_utils import tl, triton

_FP8_MIN, _FP8_MAX = get_fp8_min_max()


@triton.jit
def _quantize_pad_fp8_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_ys,
    stride_yh,
    stride_yd,
    num_heads,
    n_rows,
    n_cols,
    n_cols_padded,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    SKIP_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_rows
    mask_out = mask_m[:, None] & (offs_n[None, :] < n_cols_padded)
    mask_in = mask_m[:, None] & (offs_n[None, :] < n_cols)

    # Decompose flattened row into (token, head) for 3D stride indexing.
    s = offs_m // num_heads
    h = offs_m % num_heads

    x_ptrs = (
        x_ptr
        + s[:, None] * stride_xs
        + h[:, None] * stride_xh
        + offs_n[None, :] * stride_xd
    )
    x = tl.load(x_ptrs, mask=mask_in, other=0.0).to(tl.float32)
    if SKIP_SCALE:
        x_q = x
    else:
        scale = tl.load(scale_ptr)
        x_q = x / scale
    x_q = tl.clamp(x_q, fp8_min, fp8_max).to(y_ptr.dtype.element_ty)

    y_ptrs = (
        y_ptr
        + s[:, None] * stride_ys
        + h[:, None] * stride_yh
        + offs_n[None, :] * stride_yd
    )
    tl.store(y_ptrs, x_q, mask=mask_out)


def _get_fp8_pad_quant_config(padded_head_dim: int) -> tuple[int, int, int]:
    block_n = triton.next_power_of_2(padded_head_dim)
    block_n = max(16, min(block_n, 128))
    block_m = 16
    num_warps = 4
    return block_m, block_n, num_warps
