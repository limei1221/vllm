# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op LoRA mixin. LoRA is removed from the lean build.

All methods are no-ops when lora_config is None (which it always is here).
Kept so that GPUModelRunner can inherit without import errors.
"""

from contextlib import contextmanager
from typing import Any, TypeAlias

import numpy as np
import torch.nn as nn

from vllm.lora.request import LoRARequest
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch

InputBatch: TypeAlias = GPUInputBatch


class LoRAModelRunnerMixin:
    lora_config: Any
    get_model: Any

    def reset_lora_state(self) -> None:
        pass

    def load_lora_model(self, *args: Any, **kwargs: Any) -> nn.Module:
        return args[0] if args else None

    def _set_active_loras(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_active_loras(self, *args: Any, **kwargs: Any) -> None:
        pass

    @contextmanager
    def maybe_setup_dummy_loras(self, lora_config: Any, *args: Any, **kwargs: Any):
        yield

    @contextmanager
    def maybe_select_dummy_loras(self, lora_config: Any, *args: Any, **kwargs: Any):
        yield

    @contextmanager
    def maybe_dummy_run_with_lora(self, lora_config: Any, *args: Any, **kwargs: Any):
        yield

    def maybe_remove_all_loras(self, lora_config: Any) -> None:
        pass

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()
