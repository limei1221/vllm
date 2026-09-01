# SPDX-License-Identifier: Apache-2.0
"""No-op stubs."""

from typing import Any


class KVConnectorBase:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


class KVConnectorBase_V1:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


class KVConnectorFactory:
    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Any:
        return None


class KVConnectorMetadata:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class KVConnectorStats:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class KVOutputAggregator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


def copy_kv_blocks(*args: Any, **kwargs: Any) -> None:
    pass
