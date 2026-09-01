# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs. LoRA resolver is removed from the lean build."""

from typing import Any


class LoRAResolver:
    """No-op stub."""


class LoRAResolverRegistry:
    """No-op stub."""

    def register(self, *args: Any, **kwargs: Any) -> None:
        pass

    def resolve(self, *args: Any, **kwargs: Any) -> None:
        return None
