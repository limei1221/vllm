# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs. LoRA worker manager is removed from the lean build."""

from typing import Any


class WorkerLoRAManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def create_lora_manager(self, *args: Any, **kwargs: Any) -> Any:
        return args[0] if args else None


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    pass
