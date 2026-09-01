# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stub. KV transfer is removed from the lean build."""

from typing import Any

_KV_CONNECTOR_AGENT: Any = None

__all__ = ["_KV_CONNECTOR_AGENT"]
