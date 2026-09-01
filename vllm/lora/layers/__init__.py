# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA layer stubs. LoRA adapter support is removed from the lean build."""

from enum import Enum
from typing import Any

import torch.nn as nn


class LoRAMappingType(Enum):
    LANGUAGE = 1
    TOWER = 2
    CONNECTOR = 3


class LoRAMapping:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class BaseLayerWithLoRA(nn.Module):
    """No-op stub so isinstance checks don't break."""


def try_get_optimal_moe_lora_config(*args: Any, **kwargs: Any) -> Any:
    return None
