# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linear kernel dispatch for the fp8-only lean build."""

from typing import TypeVar

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
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
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
    CutlassInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.humming import (
    HummingFP8ScaledMMLinearKernel,
    HummingInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
    TritonInt8ScaledMMLinearKernel,
)
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)

_KernelT = TypeVar("_KernelT", bound=ScaledMMLinearKernel | MMLinearKernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=MMLinearLayerConfig)

_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [
        CutlassInt8ScaledMMLinearKernel,
        TritonInt8ScaledMMLinearKernel,
        HummingInt8ScaledMMLinearKernel,
    ],
}

_POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [
        MarlinFP8ScaledMMLinearKernel,
        FlashInferFP8ScaledMMLinearKernel,
        CutlassFP8ScaledMMLinearKernel,
        B12xTensorFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        HummingFP8ScaledMMLinearKernel,
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
        MarlinFP8ScaledMMLinearKernel,
        TritonFp8BlockScaledMMKernel,
        HummingFP8ScaledMMLinearKernel,
        BlockWiseTorchFP8ScaledMMLinearKernel,
    ],
}

_POSSIBLE_WFP8A16_KERNELS: dict[
    PlatformEnum, list[type[FP8ScaledMMLinearKernel]]
] = {
    PlatformEnum.CUDA: [
        HummingFP8ScaledMMLinearKernel,
        MarlinFP8ScaledMMLinearKernel,
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
    config: _KernelConfigT,
    compute_capability: int | None = None,
    force_kernel: type[_KernelT] | None = None,
) -> FP8ScaledMMLinearKernel:
    if config.weight_type == "fp8":
        if config.has_weight_block_scaling:
            kernel_type = choose_scaled_mm_linear_kernel(
                config,
                _POSSIBLE_FP8_BLOCK_KERNELS,
                compute_capability,
                force_kernel,
            )
        else:
            kernel_type = choose_scaled_mm_linear_kernel(
                config,
                _POSSIBLE_FP8_KERNELS,
                compute_capability,
                force_kernel,
            )
    else:
        raise ValueError(f"Unsupported weight type: {config.weight_type}")
    return kernel_type(config)  # type: ignore


def init_int8_linear_kernel(
    config: _KernelConfigT,
    compute_capability: int | None = None,
) -> Int8ScaledMMLinearKernel:
    kernel_type = choose_scaled_mm_linear_kernel(
        config,
        _POSSIBLE_INT8_KERNELS,
        compute_capability,
    )
    return kernel_type(config)  # type: ignore


def init_wfp8_a16_linear_kernel(
    config: _KernelConfigT,
    compute_capability: int | None = None,
) -> FP8ScaledMMLinearKernel:
    kernel_type = choose_scaled_mm_linear_kernel(
        config,
        _POSSIBLE_WFP8A16_KERNELS,
        compute_capability,
    )
    return kernel_type(config)  # type: ignore


__all__ = [
    "init_fp8_linear_kernel",
    "init_int8_linear_kernel",
    "init_wfp8_a16_linear_kernel",
    "FP8ScaledMMLinearKernel",
    "Int8ScaledMMLinearKernel",
    "ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "Int8ScaledMMLinearLayerConfig",
    "ScaledMMLinearLayerConfig",
    "CutlassFP8ScaledMMLinearKernel",
    "CutlassInt8ScaledMMLinearKernel",
    "FlashInferFP8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "TritonInt8ScaledMMLinearKernel",
    "_KernelT",
    "DeepGemmFp8BlockScaledMMKernel",
    "FlashInferFp8DeepGEMMDynamicBlockScaledKernel",
    "B12xFp8BlockScaledMMKernel",
    "B12xTensorFP8ScaledMMLinearKernel",
]
