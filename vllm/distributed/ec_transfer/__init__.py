# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs. EC transfer is removed from the lean build."""

import enum
from typing import Any


def get_ec_transfer(*args: Any, **kwargs: Any) -> Any:
    return None


def has_ec_transfer(*args: Any, **kwargs: Any) -> bool:
    return False


def ensure_ec_transfer_initialized(*args: Any, **kwargs: Any) -> None:
    pass


def ensure_ec_transfer_shutdown(*args: Any, **kwargs: Any) -> None:
    pass


class ECConnectorRole(enum.Enum):
    SCHEDULER = enum.auto()
    WORKER = enum.auto()


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

    @staticmethod
    def create_connector(*args: Any, **kwargs: Any) -> Any:
        return None


class ECConnectorWorkerMetadata:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ECOutputAggregator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None
