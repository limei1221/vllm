# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal support has been removed from this lean build.

The symbols below are no-op stubs kept so that downstream modules import
cleanly. ``MULTIMODAL_REGISTRY.supports_multimodal_inputs`` always returns
False, so every multimodal code path is inert.
"""

from typing import Any, TypeAlias

NestedTensors: TypeAlias = Any


def _bool_false(*args: Any, **kwargs: Any) -> bool:
    return False


def _int_zero(*args: Any, **kwargs: Any) -> int:
    return 0


def _empty_list(*args: Any, **kwargs: Any) -> list:
    return []


def _empty_dict(*args: Any, **kwargs: Any) -> dict:
    return {}


def _none(*args: Any, **kwargs: Any) -> None:
    return None


class _Permissive:
    """Object whose attribute access / call returns permissive no-ops.

    Used for stubs whose full method surface is large and only ever reached
    on inert multimodal paths. Any missing attribute resolves to a callable
    that itself returns permissive defaults, so AttributeError can't occur.
    """

    _false_attrs = {
        "supports_multimodal_inputs",
        "supports_multimodal_raw_inputs",
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._false_attrs:
            return _bool_false
        return _PermissiveCallable()

    def supports_multimodal_inputs(self, model_config: Any) -> bool:
        return False

    def supports_multimodal_raw_inputs(self, model_config: Any) -> bool:
        return False

    def get_num_mm_connector_tokens(self, model_config: Any) -> int:
        return 0

    def worker_receiver_cache_from_config(self, vllm_config: Any) -> Any:
        return None

    def register_processor(self, *args: Any, **kwargs: Any) -> None:
        pass


class _PermissiveCallable:
    """Callable that returns permissive defaults and accepts anything."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def __getattr__(self, name: str) -> Any:
        return _PermissiveCallable()


class MultiModalRegistry(_Permissive):
    pass


MULTIMODAL_REGISTRY = MultiModalRegistry()
