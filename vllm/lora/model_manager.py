# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stub. LoRA model manager is removed from the lean build."""

from typing import Any


class LoRAModelManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class LRUCacheLoRAModelManager(LoRAModelManager):
    pass


def create_lora_manager(*args: Any, **kwargs: Any) -> Any:
    return args[0] if args else None
