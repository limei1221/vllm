# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

current_platform.import_kernels()

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


# page attention ops


# merge attn states ops
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    torch.ops._C.merge_attn_states(
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefill_tokens_with_context,
        output_scale,
    )


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rope_dim_offset: int = 0,
    inverse: bool = False,
) -> None:
    if rope_dim_offset == 0 and not inverse:
        torch.ops._C.rotary_embedding(
            positions, query, key, head_size, cos_sin_cache, is_neox
        )
    else:
        torch.ops._C.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
            rope_dim_offset,
            inverse,
        )


# layer norm ops
def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


# LongCat n-gram embedding index kernel (see csrc/.../ngram_embedding_kernels.cu).


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> None:
    # Note: this func is batch invariant
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    torch.ops._C.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        forced_token_heads_per_warp,
    )


def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


def apply_repetition_penalties_cuda(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    torch.ops._C.apply_repetition_penalties_(
        logits, prompt_mask, output_mask, repetition_penalties
    )


def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    if logits.is_cuda and logits.is_contiguous():
        apply_repetition_penalties_cuda(
            logits, prompt_mask, output_mask, repetition_penalties
        )
    else:
        apply_repetition_penalties_torch(
            logits, prompt_mask, output_mask, repetition_penalties
        )


# fused quant layer norm ops
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(input.shape, dtype=quant_dtype, device=input.device)
    scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )

    torch.ops._C.rms_norm_dynamic_per_token_quant(
        output, input, weight, scales, epsilon, scale_ub, residual
    )
    return output, scales


# fused quant layer norm ops blocked
def rms_norm_per_block_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    group_size: list[int],
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
    tma_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(group_size) == 2
    output = torch.empty(input.shape, dtype=quant_dtype, device=input.device)
    if is_scale_transposed:
        if tma_alignment == 0:
            scales = torch.empty(
                (input.shape[-1] // group_size[1], input.numel() // input.shape[-1]),
                device=input.device,
                dtype=torch.float32,
            ).transpose(0, 1)
        else:
            m = input.shape[-2]
            sf_k = input.shape[-1] // group_size[1]
            tma_aligned_m = (m + tma_alignment - 1) // tma_alignment * tma_alignment
            shape = input.shape[:-2] + (m, sf_k)
            stride = (
                (1, tma_aligned_m)
                if input.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            scales = torch.empty_strided(
                shape, stride, device=input.device, dtype=torch.float32
            )
    else:
        scales = torch.empty(
            (input.numel() // input.shape[-1], input.shape[-1] // group_size[1]),
            device=input.device,
            dtype=torch.float32,
        )

    assert tma_alignment in [0, 4], "Expected TMA alignment 0 or 4, but got " + str(
        tma_alignment
    )

    torch.ops._C.rms_norm_per_block_quant(
        output,
        input,
        weight,
        scales,
        epsilon,
        scale_ub,
        residual,
        group_size[1],
        is_scale_transposed,
    )
    return output, scales


# fused silu_and_mul + block quant
def silu_and_mul_per_block_quant(
    input: torch.Tensor,
    group_size: int,  # Changed from list[int]
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim == 2, f"input must be 2D [batch, hidden*2], got {input.shape}"
    assert input.shape[-1] % 2 == 0, (
        f"input last dim must be even (gate||up layout), got {input.shape[-1]}"
    )

    # Output is half the width of input (after silu_and_mul)
    num_tokens = input.shape[0]
    hidden_size = input.shape[-1] // 2  # Divide by 2 because input is [gate || up]

    # Allocate output tensor (FP8 or INT8)
    output = torch.empty(
        (num_tokens, hidden_size), device=input.device, dtype=quant_dtype
    )

    # Allocate scales tensor
    num_groups = hidden_size // group_size  # Directly use group_size
    if is_scale_transposed:
        scales = torch.empty(
            (num_groups, num_tokens),
            device=input.device,
            dtype=torch.float32,
        ).t()
    else:
        scales = torch.empty(
            (num_tokens, num_groups),
            device=input.device,
            dtype=torch.float32,
        )

    # Call the C++ kernel
    torch.ops._C.silu_and_mul_per_block_quant(
        output,
        input,
        scales,
        group_size,  # Pass directly as int
        scale_ub,
        is_scale_transposed,
    )

    return output, scales


# quantization ops
# cutlass


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])

    cutlass_compatible_b = b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    if not cutlass_compatible_b:
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
            triton_scaled_mm,
        )

        out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    else:
        out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out.view(*target_shape)


def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool:
    if cuda_device_capability < 90 or cuda_device_capability >= 110:
        return False
    try:
        return torch.ops._C.cutlass_group_gemm_supported(cuda_device_capability)
    except AttributeError:
        # Return False on non-CUDA platforms where it is not available
        return False


# CUTLASS W4A8


if hasattr(torch.ops._C, "permute_cols"):

    @register_fake("_C::permute_cols")
    def _permute_cols_fake(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
    group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8 (must be 2D: [M, N])
        scale: Optional scaling factor for the FP8 quantization. Supports:
            - 0D or [1]: per-tensor scaling
            - 1D: requires explicit group_shape to disambiguate per-channel
              vs per-token (use (-1, 1) for per-channel, (1, -1) for per-token)
            - 2D [M/group_m, N/group_n]: group scaling (e.g. [M, N/128] for
              DeepSeek-style (1,128) groups, or [M/128, N/128] for (128,128))
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.
        group_shape: Optional tuple (group_m, group_n) specifying the group
            shape for static quantization. Use -1 for "full extent" (e.g.,
            (-1, -1) for per-tensor, (-1, 1) for per-channel, etc.)
            Required for 1D scales; optional for 2D scales.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: tuple[int, int] | torch.Size = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert num_token_padding is None, "padding not supported if output passed in"
        assert output.dtype == out_dtype

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub
            )
        else:
            scale = torch.empty(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        torch.ops._C.static_scaled_fp8_quant(output, input, scale, group_shape)

    return output, scale


# int8


# mamba


# ROCm skinny gemms
def LLMM1(a: torch.Tensor, b: torch.Tensor, rows_per_block: int) -> torch.Tensor:
    return torch.ops._rocm_C.LLMM1(a, b, rows_per_block)


def wvSplitK(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitK(a, b, bias, cu_count)


if hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "wvSplitK_int4_g"):

    @register_fake("_rocm_C::wvSplitK_int4_g")
    def _wvSplitK_int4_g_fake(
        in_a: torch.Tensor,  # packed int4 weight [out_features, K/2]
        in_b: torch.Tensor,  # activation [num_tokens, K]
        in_scale: torch.Tensor,
        in_zero_points: torch.Tensor | None,
        in_bias: torch.Tensor | None,
        CuCount: int,
        group_size: int,
    ) -> torch.Tensor:
        # Kernel returns {in_b.size(0), in_a.size(0)} = [num_tokens, out_features].
        num_tokens = in_b.size(0)
        out_features = in_a.size(0)
        return torch.empty(
            (num_tokens, out_features), dtype=in_b.dtype, device=in_b.device
        )


def wvSplitKrc(
    a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitKrc(a, b, bias, cu_count)


# moe
def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
):
    torch.ops._moe_C.moe_sum(input, output, topk_ids, expert_map)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map,
    )


def batched_moe_align_block_size(
    max_tokens_per_batch: int,
    block_size: int,
    expert_num_tokens: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    torch.ops._moe_C.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )


def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
        expert_map,
    )


def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    output = torch.empty(
        hidden_states.shape[0],
        router_weight.shape[0],
        device=hidden_states.device,
        dtype=output_dtype,
    )
    torch.ops._moe_C.dsv3_router_gemm(output, hidden_states, router_weight)
    return output


def fp32_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        hidden_states.shape[0],
        router_weight.shape[0],
        device=hidden_states.device,
        dtype=torch.float32,
    )
    torch.ops._C.fp32_router_gemm(output, hidden_states, router_weight)
    return output


if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "fp32_router_gemm"):

    @register_fake("_C::fp32_router_gemm")
    def fp32_router_gemm_fake(
        output: torch.Tensor,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
    ) -> None:
        return


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        is_padding,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    is_padding: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
        is_padding,
    )


def topk_hash_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
        is_padding,
    )


def grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    """
    Perform grouped top-k routing for mixture of experts.

    Args:
        scores: Raw inputs (logits if scoring_func=1, scores if scoring_func=0)
        num_expert_group: Number of expert groups
        topk_group: Number of groups to select
        topk: Number of experts to select per token
        renormalize: Whether to renormalize the output weights
        routed_scaling_factor: Scaling factor for routing weights
        bias: Bias tensor (e_score_correction_bias). Always fused in kernel.
        scoring_func: 0=none (no activation), 1=sigmoid
    """
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The fused grouped_topk kernel is only available on CUDA platforms"
        )
    return torch.ops._moe_C.grouped_topk(
        scores,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )


def concat_and_cache_mla_rope_fused(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    kv_cache_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla_rope_fused(
        positions,
        q_pe,
        k_pe,
        kv_c,
        cos_sin_cache,
        is_neox,
        slot_mapping,
        kv_cache,
        kv_cache_dtype,
        kv_cache_scale,
    )


def convert_fp8(
    output: torch.Tensor, input: torch.Tensor, scale: float = 1.0, kv_dtype: str = "fp8"
) -> None:
    torch.ops._C_cache_ops.convert_fp8(output, input, scale, kv_dtype)


def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts,
    )


def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    torch.ops._C_cache_ops.cp_gather_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts
    )


def cp_gather_and_upconvert_fp8_kv_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    workspace_starts: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Gather and upconvert FP8 KV cache to BF16 workspace.

    Args:
        src_cache: FP8 KV cache [num_blocks, block_size, 656]
        dst: BF16 output workspace [total_tokens, 576]
        block_table: Block indices [num_reqs, max_blocks]
        workspace_starts: Workspace start offsets [num_reqs]
        batch_size: Number of requests
        seq_starts: Optional source sequence offsets [num_reqs]
    """
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        src_cache, dst, block_table, workspace_starts, batch_size, seq_starts
    )


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None:
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype
    )


def top_k_per_row_prefill(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    torch.ops._C.top_k_per_row_prefill(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    )


def top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    )


def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # ruff: noqa: E501
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device
    )


# custom ar
def init_custom_ar(
    ipc_tensors: list[torch.Tensor],
    rank_data: torch.Tensor,
    rank: int,
    fully_connected: bool,
) -> int:
    return torch.ops._C_custom_ar.init_custom_ar(
        ipc_tensors, rank_data, rank, fully_connected
    )


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)


def custom_all_gather(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.custom_all_gather(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes
    )


def mnnvl_lamport_all_gather(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    local_buffer: int,
    multicast_buffer: int,
    epoch_buffer: int,
    stage_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.mnnvl_lamport_all_gather(
        fa,
        inp,
        out,
        local_buffer,
        multicast_buffer,
        epoch_buffer,
        stage_sz_bytes,
    )


def custom_reduce_scatter(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.custom_reduce_scatter(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes
    )


def mnnvl_lamport_reduce_scatter(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    local_buffer: int,
    epoch_buffer: int,
    stage_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.mnnvl_lamport_reduce_scatter(
        fa, inp, out, local_buffer, epoch_buffer, stage_sz_bytes
    )


def dispose(fa: int) -> None:
    torch.ops._C_custom_ar.dispose(fa)


def meta_size() -> int:
    return torch.ops._C_custom_ar.meta_size()


def register_buffer(fa: int, ipc_tensors: list[int]) -> None:
    return torch.ops._C_custom_ar.register_buffer(fa, ipc_tensors)


def get_graph_buffer_ipc_meta(fa: int) -> tuple[list[int], list[int]]:
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(
    fa: int, handles: list[list[int]], offsets: list[list[int]]
) -> None:
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)


def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]:
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)


def open_mem_handle(mem_handle: torch.Tensor):
    return torch.ops._C_custom_ar.open_mem_handle(mem_handle)


def free_shared_buffer(ptr: int) -> None:
    torch.ops._C_custom_ar.free_shared_buffer(ptr)


# quick all reduce


def dsv3_fused_a_gemm(
    output: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    enable_pdl: bool = False,
) -> None:
    """Low-latency fused-A-style GEMM (SM 9.0+, BF16, 1-16 tokens).

    Computes ``output = mat_a @ mat_b`` for the compiled Kimi K3 and
    DeepSeek V3 projection shapes. ``mat_a`` and ``output`` are row-major;
    ``mat_b`` is the column-major transposed weight. ``enable_pdl`` permits
    programmatic dependent launch for callers that have validated it.

    Requires SM 9.0+.
    """
    torch.ops._C.dsv3_fused_a_gemm(output, mat_a, mat_b, enable_pdl)


def moe_wna16_gemm(*args, **kwargs) -> None:
    raise NotImplementedError("W4A16/W8A16 MoE kernels are not available in this build")


def hadacore_transform(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    Perform Hadamard transforms using [Hadacore](https://arxiv.org/abs/2412.08832)
    kernels. Note that these kernels exploit the recursive properties of
    Sylvester Hadamards, and therefore do not require transform weight data

    Note that sylvester hadamard transforms are also symmetric, which means that
    this function is also applies the (transpose <=> inverse) transform.

    Args:
        x: value to be transformed inplace
        inplace: modify value in place

    Returns:
        value after transformation
    """
    return torch.ops._C.hadacore_transform(x, inplace)


if hasattr(torch.ops._C, "hadacore_transform"):

    @register_fake("_C::hadacore_transform")
    def _hadacore_transform_fake(x: torch.Tensor, inplace: bool) -> torch.Tensor:
        return torch.empty_like(x) if not inplace else x


if hasattr(torch.ops._C, "minimax_allreduce_rms_qk"):

    @register_fake("_C::minimax_allreduce_rms_qk")
    def _minimax_allreduce_rms_qk_fake(
        qkv: torch.Tensor,
        norm_weight_q: torch.Tensor,
        norm_weight_k: torch.Tensor,
        workspace: torch.Tensor,
        q_size: int,
        kv_size: int,
        rank: int,
        nranks: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_num = qkv.shape[0]
        return (
            torch.empty([token_num, q_size], dtype=qkv.dtype, device=qkv.device),
            torch.empty([token_num, kv_size], dtype=qkv.dtype, device=qkv.device),
        )
