# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs."""

from vllm.distributed.ec_transfer import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
    ECConnectorWorkerMetadata,
)

__all__ = [
    "ECConnectorBase",
    "ECConnectorMetadata",
    "ECConnectorRole",
    "ECConnectorWorkerMetadata",
]
