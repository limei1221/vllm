# SPDX-License-Identifier: Apache-2.0
from typing import Any

_ProcessorFactories = Any


class MultiModalTimingRegistry:
    """No-op stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self
