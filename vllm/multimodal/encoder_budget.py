# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op. Multimodal encoder budgeting is removed from the lean build."""

from typing import Any

from vllm.multimodal import _PermissiveCallable


class MultiModalBudget:
    """No-op budget stub; any attribute access returns a permissive no-op."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return _PermissiveCallable()


def get_dummy_encoder_profile_inputs(*args: Any, **kwargs: Any) -> list:
    return []
