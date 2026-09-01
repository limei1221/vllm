# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs. KV transfer is removed from the lean build."""

from typing import Any

from vllm.distributed.kv_transfer import kv_transfer_state


def get_kv_transfer_group(*args: Any, **kwargs: Any) -> Any:
    return None


def has_kv_transfer_group(*args: Any, **kwargs: Any) -> bool:
    return False


def is_v1_kv_transfer_group(*args: Any, **kwargs: Any) -> bool:
    return False


def ensure_kv_transfer_initialized(*args: Any, **kwargs: Any) -> None:
    pass


def ensure_kv_transfer_shutdown(*args: Any, **kwargs: Any) -> None:
    pass


__all__ = [
    "ensure_kv_transfer_initialized",
    "ensure_kv_transfer_shutdown",
    "get_kv_transfer_group",
    "has_kv_transfer_group",
    "is_v1_kv_transfer_group",
    "kv_transfer_state",
]
