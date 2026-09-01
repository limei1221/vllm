# SPDX-License-Identifier: Apache-2.0
"""No-op stubs. KV transfer is removed from the lean build."""

from typing import Any


def get_kv_transfer_group(*args: Any, **kwargs: Any) -> Any:
    return None


def has_kv_transfer_group(*args: Any, **kwargs: Any) -> bool:
    return False
