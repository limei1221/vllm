# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs. Weight transfer is removed from the lean build."""

from typing import Any, Generic, TypeVar


class WeightTransferManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


_InitInfoT = TypeVar("_InitInfoT")
_UpdateInfoT = TypeVar("_UpdateInfoT")


class BaseWeightTransferEngine(Generic[_InitInfoT, _UpdateInfoT]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


WeightTransferEngine = BaseWeightTransferEngine


class WeightTransferEngineFactory:
    @staticmethod
    def create_engine(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Weight transfer is not supported by this build.")


class WeightTransferInitInfo:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class WeightTransferUpdateInfo:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class WeightTransferInitRequest:
    init_info: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class WeightTransferUpdateRequest:
    update_info: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


__all__ = [
    "BaseWeightTransferEngine",
    "WeightTransferInitInfo",
    "WeightTransferUpdateInfo",
    "WeightTransferEngine",
    "WeightTransferEngineFactory",
    "WeightTransferInitRequest",
    "WeightTransferManager",
    "WeightTransferUpdateRequest",
]
