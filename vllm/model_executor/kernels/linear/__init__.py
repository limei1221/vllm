# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linear kernel dispatch for the fp8-only lean build."""

from typing import TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.base import (
    MMLinearKernel,
    MMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x_block import (
    B12xFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor import (
    B12xTensorFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)

_KernelT = TypeVar("_KernelT", bound=ScaledMMLinearKernel | MMLinearKernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=MMLinearLayerConfig)

_POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [
        FlashInferFP8ScaledMMLinearKernel,
        CutlassFP8ScaledMMLinearKernel,
        B12xTensorFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
}

_POSSIBLE_FP8_BLOCK_KERNELS: dict[
    PlatformEnum,
    list[type[Fp8BlockScaledMMLinearKernel | FP8ScaledMMLinearKernel]],
] = {
    PlatformEnum.CUDA: [
        FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
        DeepGemmFp8BlockScaledMMKernel,
        CutlassFp8BlockScaledMMKernel,
        B12xFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
        BlockWiseTorchFP8ScaledMMLinearKernel,
    ],
}


def is_supported_and_can_implement_kernel(
    kernel: type[_KernelT],
    config: _KernelConfigT,
    compute_capability: int | None,
) -> tuple[bool, str]:
    if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
        return False, f" {kernel.__name__} is disabled by environment variable"

    if compute_capability is None:
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc[0] * 10 + _cc[1]

    is_supported, failure_reason = kernel.is_supported(compute_capability)
    if not is_supported:
        return False, f"{kernel.__name__} {failure_reason}."

    can_implement, failure_reason = kernel.can_implement(config)
    if not can_implement:
        return False, f"{kernel.__name__} {failure_reason}."

    return True, ""


def choose_scaled_mm_linear_kernel(
    config: _KernelConfigT,
    possible_kernels: dict[PlatformEnum, list[type[_KernelT]]],
    compute_capability: int | None = None,
    force_kernel: type[_KernelT] | None = None,
) -> type[_KernelT]:
    failure_reason_list = []

    if force_kernel is not None:
        can_implement, failure_reason = is_supported_and_can_implement_kernel(
            force_kernel, config, compute_capability
        )
        if can_implement:
            return force_kernel
        failure_reason_list.append(failure_reason)

    platform_kernels = possible_kernels.get(current_platform.platform_enum, [])
    for kernel_type in platform_kernels:
        can_implement, failure_reason = is_supported_and_can_implement_kernel(
            kernel_type, config, compute_capability
        )
        if can_implement:
            return kernel_type
        failure_reason_list.append(failure_reason)

    raise ValueError(
        f"No supported linear kernel found for config:\n{config}\n"
        f"Failure reasons: {failure_reason_list}"
    )


def init_fp8_linear_kernel(
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
    input_dtype: torch.dtype,
    out_dtype: torch.dtype,
    weight_shape: tuple[int, int],
    compute_capability: int | None = None,
    force_kernel: type[FP8ScaledMMLinearKernel] | None = None,
    module_name: str | None = None,
) -> FP8ScaledMMLinearKernel | Fp8BlockScaledMMLinearKernel:
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        input_dtype=input_dtype,
        out_dtype=out_dtype,
        weight_shape=weight_shape,
    )

    # Per-group activation scales mean block-wise quantization, which has its
    # own kernel set (DeepGEMM and friends). The two lists have different
    # element types, so they are dispatched separately to stay inferable.
    if activation_quant_key.scale.group_shape.is_per_group():
        kernel_type = choose_scaled_mm_linear_kernel(
            config,
            _POSSIBLE_FP8_BLOCK_KERNELS,  # type: ignore[misc]
            compute_capability,
            force_kernel,
        )
    else:
        kernel_type = choose_scaled_mm_linear_kernel(
            config,
            _POSSIBLE_FP8_KERNELS,  # type: ignore[arg-type]
            compute_capability,
            force_kernel,
        )

    if module_name:
        logger.info_once(
            "Selected %s for %s", kernel_type.__name__, module_name, scope="global"
        )

    # TODO make scaled_mm kernels inherit from MMLinearKernel. Only the
    # FP8ScaledMMLinearKernel subclasses take layer_param_names; the block
    # kernels are constructed from the config alone.
    if issubclass(kernel_type, FP8ScaledMMLinearKernel):
        return kernel_type(
            config,
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_scale_ub",
            ],
        )

    return kernel_type(config)


__all__ = [
    "init_fp8_linear_kernel",
    "FP8ScaledMMLinearKernel",
    "ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "ScaledMMLinearLayerConfig",
    "CutlassFP8ScaledMMLinearKernel",
    "FlashInferFP8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "_KernelT",
    "DeepGemmFp8BlockScaledMMKernel",
    "FlashInferFp8DeepGEMMDynamicBlockScaledKernel",
    "B12xFp8BlockScaledMMKernel",
    "B12xTensorFP8ScaledMMLinearKernel",
]
