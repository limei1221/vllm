# SPDX-License-Identifier: Apache-2.0
"""No-op stubs. EC transfer is removed from the lean build."""

from typing import Any


def get_ec_transfer(*args: Any, **kwargs: Any) -> Any:
    return None


def has_ec_transfer(*args: Any, **kwargs: Any) -> bool:
    return False


class ECConnectorBase:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


class ECConnectorMetadata:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ECConnectorFactory:
    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Any:
        return None


class ECConnectorWorkerMetadata:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ECOutputAggregator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None
