# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

    @staticmethod
    def create_connector(*args: Any, **kwargs: Any) -> Any:
        return None

    @staticmethod
    def get_connector_class(*args: Any, **kwargs: Any) -> Any:
        return None

    @staticmethod
    def supports_hma_config(*args: Any, **kwargs: Any) -> bool:
        return False


class KVConnectorMetadata:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class KVConnectorStats:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.data: dict[str, Any] = {}

    def is_empty(self) -> bool:
        return True

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        return other


class KVOutputAggregator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


def copy_kv_blocks(*args: Any, **kwargs: Any) -> None:
    pass


def get_kv_connector_cache_layout(*args: Any, **kwargs: Any) -> Any:
    return None


class KVConnectorHandshakeMetadata:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class KVConnectorWorkerMetadata:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class KVConnectorLogging:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


class KVConnectorProm:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None
