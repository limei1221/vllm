# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from math import prod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    per_tensor_dequantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts,
    topk_numel,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    curr_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ids_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for x in range(tl.cdiv(topk_numel, BLOCK_SIZE)):
        mask = offsets < (topk_numel - x * BLOCK_SIZE)
        expert_ids = tl.load(topk_ids_ptrs, mask=mask, other=-1)
        if HAS_EXPERT_MAP:
            expert_map_ptrs = expert_map + expert_ids
            expert_map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptrs, mask=expert_map_mask, other=-1)

        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
        topk_ids_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    """
    Count the number to tokens assigned to each expert.

    Parameters:
    - topk_ids (torch.Tensor): Tensor mapping each token to its
    list of experts.
    - num_local_experts (int): Number of experts in this rank.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
    from the global expert space to the local expert space of the expert
    parallel shard.

    Returns:
    A tensor of size num_local_experts, where tensor[i] holds the number
    of tokens assigned to the ith expert.
    """
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_num_tokens


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel(), (
        f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"
    )  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    per_act_token: bool,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    if block_shape is None:
        # TODO(luka): use QuantFP8 custom op
        #  https://github.com/vllm-project/vllm/issues/20711
        A, A_scale = ops.scaled_fp8_quant(
            A, A_scale, use_per_token_if_dynamic=per_act_token
        )
    else:
        assert not per_act_token
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_fp8(A, block_k)
        assert cdiv(A.size(-1), block_k) == A_scale.size(-1)

    return A, A_scale


def _fp8_quantize_dequantize(
    A: torch.Tensor,
    A_scale: torch.Tensor,
):
    qA, qA_scale = ops.scaled_fp8_quant(A, A_scale, use_per_token_if_dynamic=False)
    A = per_tensor_dequantize(qA, qA_scale).to(A.dtype)

    return A, None


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: torch.Tensor | None,
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: list[int] | None = None,
    is_scale_swizzled: bool = True,
    ocp_mx_scheme: str | None = None,
    quantization_emulation: bool = False,
    mx_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Handle OCP MX scheme that requires QDQ (quantize-dequantize) for emulation
    if ocp_mx_scheme is not None:
        if ocp_mx_scheme in {"w_mxfp4", "w_mxfp4_a_mxfp4"}:
            pass  # No QDQ needed for these schemes
        elif ocp_mx_scheme.endswith("a_fp8"):
            # Perform QDQ (quantize and dequantize) on activation for emulation
            # purpose, because there is no native kernel for weight in ocp_mx_scheme
            # and activation in FP8. The implementation is based on existing
            # non-emulation ops.
            # TODO: Remove this `ocp_mx_scheme is not None` block and rely solely
            # on `quantization_emulation`.
            return _fp8_quantize_dequantize(A, A_scale)
        # else: For other schemes (e.g., *_a_mxfp6_e3m2, *_a_mxfp6_e2m3),
        # weights are already dequantized, and we proceed with normal
        # activation quantization below.
    if quant_dtype == current_platform.fp8_dtype():
        if quantization_emulation:
            return _fp8_quantize_dequantize(A, A_scale)
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)

    if quant_dtype is not None:
        raise NotImplementedError(
            f"MoE quantization dtype {quant_dtype!r} is not part of this build; "
            "only fp8 is supported."
        )

    return A, A_scale


def normalize_scales_shape(scales: torch.Tensor | None) -> torch.Tensor | None:
    if scales is not None:
        if scales.numel() == 1:
            scales = scales.view(1, 1)
        else:
            scales = scales.view(-1, scales.size(-1))
    return scales


@triton.jit
def _pack_topk_ids_weights_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,  # triton metadata
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    expert_id_shifted = expert_id << 16

    weight = tl.load(topk_weights_ptr + offsets, mask=mask, other=0.0)
    weight_bf16 = weight.to(tl.bfloat16)
    weight_int16 = weight_bf16.to(tl.int16, bitcast=True)

    weight_int32 = weight_int16.to(tl.int32) & 0xFFFF

    packed = expert_id_shifted | weight_int32
    tl.store(output_ptr + offsets, packed, mask=mask)


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _swiglu_limit_torch(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d]
    up = input[:, d:]

    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    output.copy_(F.silu(gate) * up)


@triton.jit
def _swiglu_limit_pad_aware_kernel(
    input_ptr,  # [num_tokens, 2 * hidden_size]
    output_ptr,  # [num_tokens, hidden_size]
    topk_ids_ptr,  # [num_tokens, num_topk]
    expert_map_ptr,  # global -> local expert id, or -1 if non-local
    hidden_size,
    input_row_stride,
    num_tokens,
    swiglu_limit,
    HAS_LIMIT: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Persistent over rows: each CTA owns one column tile and processes a
    # strided set of token assignments.
    pid = tl.program_id(0)
    row_stride = tl.num_programs(0)
    column_tile = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = column_tile < hidden_size

    for row in tl.range(pid, num_tokens, row_stride):
        expert_id = tl.load(topk_ids_ptr + row)
        should_compute = expert_id != -1
        if HAS_EXPERT_MAP:
            local_expert_id = tl.load(
                expert_map_ptr + expert_id,
                mask=expert_id >= 0,
                other=-1,
            )
            should_compute = should_compute & (local_expert_id != -1)

        if should_compute:
            gate_offsets = row.to(tl.int64) * input_row_stride + column_tile
            up_offsets = gate_offsets + hidden_size

            gate = tl.load(input_ptr + gate_offsets, mask=mask, other=0.0).to(
                tl.float32
            )

            up = tl.load(input_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)

            if HAS_LIMIT:
                gate = tl.minimum(gate, swiglu_limit)
                up = tl.maximum(up, -swiglu_limit)
                up = tl.minimum(up, swiglu_limit)

            silu_gate = gate / (1.0 + tl.exp(-gate))
            result = silu_gate * up
            tl.store(
                output_ptr + row.to(tl.int64) * hidden_size + column_tile,
                result.to(output_ptr.dtype.element_ty),
                mask=mask,
            )


def _swiglu_limit_pad_aware(
    output: torch.Tensor,
    input: torch.Tensor,
    topk_ids: torch.Tensor,
    swiglu_limit: float,
    expert_map: torch.Tensor | None = None,
) -> None:
    num_tokens, gate_up_size = input.shape
    hidden_size = gate_up_size // 2
    if num_tokens == 0:
        return

    BLOCK_SIZE = 1024
    grid = (min(num_tokens, 256), triton.cdiv(hidden_size, BLOCK_SIZE))
    _swiglu_limit_pad_aware_kernel[grid](
        input,
        output,
        topk_ids,
        expert_map,
        hidden_size,
        gate_up_size,
        num_tokens,
        swiglu_limit,
        HAS_LIMIT=swiglu_limit > 0,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )


def swiglu_limit_func(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> None:
    # The pad-aware Triton kernel skips unrouted token slots (topk_ids == -1)
    # and, when expert_map is given, slots routed to non-local experts, so it
    # requires topk_ids. Fall back to the torch implementation otherwise.
    if topk_ids is not None:
        _swiglu_limit_pad_aware(output, input, topk_ids, swiglu_limit, expert_map)
    else:
        _swiglu_limit_torch(output, input, swiglu_limit)


@functools.lru_cache
def enable_swap_ab(BLOCK_SIZE_M: int, BLOCK_SIZE_N: int) -> bool:
    return (
        current_platform.is_device_capability(90)
        and BLOCK_SIZE_M < 64
        and BLOCK_SIZE_N >= 64
    )


def moe_use_td_hw_supported() -> bool:
    """Whether the current device can run the TD (gather) path of
    ``fused_moe_kernel`` (ignores the ``VLLM_TRITON_USE_TD`` override).

    The A-load uses ``tensor_descriptor.gather``, which lowers to the PTX
    ``tile::gather4`` instruction. That instruction is part of the
    ``tcgen05``/Tensor Memory (TMEM) family introduced with Blackwell and has
    no Hopper (sm90) equivalent -- ptxas rejects it there ("Feature
    '.tile::gather4 ...' requires .target sm_100 or higher"). Unlike
    ``scatter4``, ``gather4`` is supported across the whole sm100+ range
    including consumer Blackwell (sm120/sm121): see triton-lang/triton#8498,
    which enables ``gather4`` on sm120/sm121 while leaving ``scatter4``
    unsupported there. So this gates on a blanket ``has_device_capability(100)``
    rather than the sm100 *family* check used for the scatter store path.
    """
    if current_platform.is_xpu():
        return True
    if current_platform.is_cuda():
        return current_platform.has_device_capability(100)
    return False


def resolve_moe_use_td() -> bool:
    """Tri-state resolver for ``VLLM_TRITON_USE_TD``.

    Unset auto-selects the TD path on XPU only, mirroring the attention
    dispatcher in ``triton_attn.py``. ``1``/``0`` force it on/off regardless
    of hardware; forcing ``1`` where it cannot compile (see
    ``moe_use_td_hw_supported``) fails at ptxas. Blackwell CUDA (sm100+) can
    compile it but is opt-in only, pending validation.
    """
    override = envs.VLLM_TRITON_USE_TD
    if override is None:
        return current_platform.is_xpu()
    return override


_warned_moe_use_td_ineffective = False


def warn_if_moe_use_td_ineffective(
    active_backend: str, is_quantized: bool = False
) -> None:
    """One-shot warning when ``VLLM_TRITON_USE_TD`` is set but ignored.

    Fires when the user set the env explicitly and either (a) the active
    MoE backend is not the fused Triton kernel, or (b) the model is
    quantized (the TD path falls back to the pointer path under any
    quantization).
    """
    global _warned_moe_use_td_ineffective
    if _warned_moe_use_td_ineffective:
        return
    if envs.VLLM_TRITON_USE_TD is None:
        return
    is_triton = active_backend.upper() == "TRITON"
    if is_triton and not is_quantized:
        return
    if not is_triton:
        reason = (
            f"the active MoE backend is {active_backend!r}; pass "
            "`--moe-backend triton` to enable the tensor-descriptor path"
        )
    else:
        reason = (
            "the model uses quantized MoE weights; the TD path is "
            "currently restricted to non-quantized weights and falls "
            "back to the pointer path"
        )
    logger.warning(
        "VLLM_TRITON_USE_TD is set to %s but %s.",
        envs.VLLM_TRITON_USE_TD,
        reason,
    )
    _warned_moe_use_td_ineffective = True
