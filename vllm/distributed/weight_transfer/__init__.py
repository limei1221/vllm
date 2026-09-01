# SPDX-License-Identifier: Apache-2.0
"""No-op stubs. Weight transfer is removed from the lean build."""

from typing import Any


class WeightTransferManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


class BaseWeightTransferEngine:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None
