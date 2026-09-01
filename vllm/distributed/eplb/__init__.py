# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs. EPLB is removed from the lean build."""

from typing import Any


class EplbState:
    """No-op EPLB state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def maybe_register_model(self, *args: Any, **kwargs: Any) -> bool:
        return False

    @classmethod
    def from_mapping(cls, *args: Any, **kwargs: Any) -> "EplbState":
        return cls()

    @classmethod
    def build_initial_global_physical_to_logical_map(
        cls, *args: Any, **kwargs: Any
    ) -> Any:
        return None

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


class EplbLayerState:
    """No-op EPLB layer state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


def override_envs_for_eplb(*args: Any, **kwargs: Any) -> None:
    pass
