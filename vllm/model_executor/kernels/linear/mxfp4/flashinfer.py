# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutedsl

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig


def swizzle_mxfp4_scales(
    scales: torch.Tensor,
    N: int,
    K: int,
) -> torch.Tensor:
    """Swizzle flat [N, K//32] E8M0 scales to CUTLASS tiled layout.

    CUTLASS expects MX scale factors in a tiled layout:
        [numMTiles, numKTiles, 32, 4, 4]
    where numMTiles = ceil(N/128), numKTiles = ceil(K/128),
    and the inner dimensions correspond to the swizzle pattern:
        mTileIdx = mIdx / 128
        outerMIdx = mIdx % 32
        innerMIdx = (mIdx / 32) % 4
        kTileIdx = kIdx / 4
        innerKIdx = kIdx % 4
    with kIdx = col_in_scale_space (i.e., index into K//32).
    """
    assert scales.dtype == torch.uint8
    num_scale_cols = K // 32  # number of E8M0 scale values per row

    num_m_tiles = (N + 127) // 128
    num_k_tiles = (num_scale_cols + 3) // 4

    # Pad N to multiple of 128 and scale_cols to multiple of 4
    padded_N = num_m_tiles * 128
    padded_scale_cols = num_k_tiles * 4

    # Start with flat scales, pad if needed
    padded = torch.zeros(
        padded_N, padded_scale_cols, dtype=torch.uint8, device=scales.device
    )
    padded[:N, :num_scale_cols] = scales

    # Reshape to tile structure:
    # [numMTiles, 4, 32, numKTiles, 4]
    #  mTileIdx, innerMIdx, outerMIdx, kTileIdx, innerKIdx
    tiled = padded.reshape(num_m_tiles, 4, 32, num_k_tiles, 4)
    # Permute to [numMTiles, numKTiles, 32, 4, 4]
    #            (outerMIdx, innerMIdx, innerKIdx)
    tiled = tiled.permute(0, 3, 2, 1, 4).contiguous()
    return tiled.reshape(-1)


_MXFP4_GROUP_SIZE = 32


class FlashInferMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 W4A4 GEMM via FlashInfer CUTLASS (SM100+)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.has_device_capability(100) and has_flashinfer_cutedsl():
            return True, None
        return False, "FlashInfer + >=sm_100 (Blackwell) required"

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "only supports MXFP4 dynamic activation"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        N, scale_K = layer.weight_scale.shape
        K = scale_K * _MXFP4_GROUP_SIZE

        # swizzle pads N to the next multiple of 128 for CUTLASS tiling
        padded_N = ((N + 127) // 128) * 128
        layer.weight_scale = Parameter(
            swizzle_mxfp4_scales(layer.weight_scale.data, N, K).reshape(padded_N, -1),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.flashinfer import (
            flashinfer_mxfp4_quantize,
            flashinfer_scaled_fp4_mm,
        )

        weight = layer.weight
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)
        x_2d = x.reshape(-1, x.shape[-1])

        x_fp4, x_scale = flashinfer_mxfp4_quantize(
            x_2d.contiguous(), backend="cute-dsl"
        )
        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            weight,
            x_scale,
            layer.weight_scale,
            alpha=None,
            out_dtype=x.dtype,
            backend="cute-dsl",
            block_size=_MXFP4_GROUP_SIZE,
            use_nvfp4=False,
        )

        if bias is not None:
            out = out + bias
        return out.view(out_shape)
