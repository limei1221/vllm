# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.safetensors_loader import (
    SafetensorsModelLoader,
)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name,
    get_model_architecture,
    get_model_cls,
)

LoadFormats = Literal["safetensors"]


def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
) -> nn.Module:
    if model_config is None:
        model_config = vllm_config.model_config
    return SafetensorsModelLoader(vllm_config.load_config).load_model(
        vllm_config=vllm_config,
        model_config=model_config,
        prefix=prefix,
    )


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get the safetensors model loader."""
    return SafetensorsModelLoader(load_config)


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "BaseModelLoader",
    "SafetensorsModelLoader",
]
