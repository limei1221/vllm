# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs."""

import enum
from typing import Any

from vllm.distributed.kv_transfer.kv_connector import (
    KVConnectorBase_V1,
    KVConnectorFactory,
    KVConnectorMetadata,
)


class KVConnectorRole(enum.Enum):
    SCHEDULER = enum.auto()
    WORKER = enum.auto()


class SupportsHMA:
    """Marker protocol; no connector in this build implements it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def request_finished_all_groups(self, *args: Any, **kwargs: Any) -> Any:
        return None


__all__ = [
    "KVConnectorBase_V1",
    "KVConnectorFactory",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "SupportsHMA",
]
