# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs."""

from vllm.distributed.weight_transfer import (
    BaseWeightTransferEngine,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferInitRequest,
    WeightTransferManager,
    WeightTransferUpdateInfo,
    WeightTransferUpdateRequest,
)

__all__ = [
    "BaseWeightTransferEngine",
    "WeightTransferEngine",
    "WeightTransferInitInfo",
    "WeightTransferInitRequest",
    "WeightTransferManager",
    "WeightTransferUpdateInfo",
    "WeightTransferUpdateRequest",
]
