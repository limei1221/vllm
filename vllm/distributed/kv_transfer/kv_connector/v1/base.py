# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs."""

from vllm.distributed.kv_transfer.kv_connector import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

__all__ = [
    "KVConnectorHandshakeMetadata",
    "KVConnectorMetadata",
    "KVConnectorWorkerMetadata",
]
