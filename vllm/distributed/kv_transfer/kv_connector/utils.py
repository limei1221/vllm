# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs."""

from vllm.distributed.kv_transfer.kv_connector import (
    KVConnectorFactory,
    KVOutputAggregator,
    copy_kv_blocks,
    get_kv_connector_cache_layout,
)

__all__ = [
    "KVConnectorFactory",
    "KVOutputAggregator",
    "copy_kv_blocks",
    "get_kv_connector_cache_layout",
]
