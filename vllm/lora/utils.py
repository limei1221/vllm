# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal LoRA utils stubs. LoRA is removed from the lean build."""

from typing import Any


def get_captured_lora_counts(*args: Any, **kwargs: Any) -> list[int]:
    return [0]


def get_adapter_absolute_path(path: str) -> str:
    return path


def create_lora_manager(*args: Any, **kwargs: Any) -> Any:
    return args[0] if args else None
