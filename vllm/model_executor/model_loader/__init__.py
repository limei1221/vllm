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
    load_config: LoadConfig | None = None,
) -> nn.Module:
    """Load a model with the safetensors loader.

    Args:
        vllm_config: The engine configuration.
        model_config: Model to load; defaults to the engine's own model.
        prefix: Weight-name prefix for the loaded module.
        load_config: Loader settings; defaults to the engine's own. Draft
            models pass their own so they resolve their own checkpoint.

    Returns:
        The loaded model.
    """
    if model_config is None:
        model_config = vllm_config.model_config
    return SafetensorsModelLoader(load_config or vllm_config.load_config).load_model(
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
