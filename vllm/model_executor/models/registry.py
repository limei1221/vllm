# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused model registry for the DeepSeek-only build."""

from __future__ import annotations

import importlib
from collections.abc import Set
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

SUPPORTED_MODELS: dict[str, tuple[str, str]] = {
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "DeepSeekMTPModel": ("deepseek_mtp", "DeepSeekMTP"),
    "EagleDeepSeekMTPModel": ("deepseek_eagle", "EagleDeepseekV3ForCausalLM"),
    "Eagle3DeepseekV2ForCausalLM": (
        "deepseek_eagle3",
        "Eagle3DeepseekV2ForCausalLM",
    ),
    "Eagle3DeepseekV3ForCausalLM": (
        "deepseek_eagle3",
        "Eagle3DeepseekV2ForCausalLM",
    ),
}

SUPPORTED_ARCHITECTURES = frozenset(SUPPORTED_MODELS.keys())


def resolve_deepseek_model_class(architecture: str) -> type[nn.Module]:
    """Resolve an architecture name to its model class.

    Args:
        architecture: The HuggingFace architectures string.

    Returns:
        The model class implementing the architecture.

    Raises:
        ValueError: If the architecture is not a supported DeepSeek variant.
    """
    try:
        module_name, class_name = SUPPORTED_MODELS[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported architecture {architecture!r}; expected DeepSeek V2/V3"
        ) from exc
    module = importlib.import_module(f"vllm.model_executor.models.{module_name}")
    return getattr(module, class_name)


@dataclass
class _ModelInfo:
    architecture: str
    is_text_generation_model: bool = True
    is_pooling_model: bool = False
    supports_pp: bool = True
    has_inner_state: bool = False
    is_attention_free: bool = False
    is_hybrid: bool = False
    has_noops: bool = False
    supports_mamba_prefix_caching: bool = False
    supports_replayssm: bool = False
    supports_transcription: bool = False
    supports_transcription_only: bool = False
    requires_raw_input_tokens: bool = False


@dataclass
class _ModelRegistry:
    """Registry of supported model architectures."""

    models: dict[str, type[nn.Module]] = field(default_factory=dict)

    def get_supported_archs(self) -> Set[str]:
        return SUPPORTED_ARCHITECTURES

    def _normalize_arch(
        self,
        architecture: str,
        model_config: ModelConfig | None = None,
    ) -> str:
        """Return the registered name for ``architecture``.

        The focused registry keys on the HuggingFace architecture name
        directly, so no remapping is needed.
        """
        del model_config
        return architecture

    def register_model(
        self,
        model_arch: str,
        model_cls: type[nn.Module] | str,
    ) -> None:
        """Register a model class for an architecture.

        Args:
            model_arch: The architecture name.
            model_cls: The model class or a ``"module:Class"`` string.

        Raises:
            TypeError: If the arguments have wrong types.
        """
        if not isinstance(model_arch, str):
            raise TypeError(
                f"`model_arch` should be a string, not a {type(model_arch)}"
            )
        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                raise ValueError("Expected a string in the format `<module>:<class>`")
            module = importlib.import_module(split_str[0])
            model_cls = getattr(module, split_str[1])
        if not (isinstance(model_cls, type) and issubclass(model_cls, nn.Module)):
            raise TypeError(
                f"`model_cls` should be a nn.Module subclass, not {type(model_cls)}"
            )
        self.models[model_arch] = model_cls

    def _resolve(self, architectures: str | list[str]) -> tuple[type[nn.Module], str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")
        for arch in architectures:
            if arch in self.models:
                return self.models[arch], arch
            if arch in SUPPORTED_MODELS:
                cls = resolve_deepseek_model_class(arch)
                self.models[arch] = cls
                return cls, arch
        raise ValueError(
            f"Model architectures {architectures} are not supported. "
            f"Supported: {SUPPORTED_ARCHITECTURES}"
        )

    def resolve_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> tuple[type[nn.Module], str]:
        return self._resolve(architectures)

    def inspect_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> tuple[_ModelInfo, str]:
        _, arch = self._resolve(architectures)
        return _ModelInfo(architecture=arch), arch

    def is_text_generation_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return True

    def is_pooling_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return False

    def is_pp_supported_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return True

    def model_has_inner_state(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return False

    def is_attention_free_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return False

    def is_hybrid_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return False

    def is_noops_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return False

    def is_transcription_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return False

    def is_transcription_only_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig | None = None,
    ) -> bool:
        return False


ModelRegistry = _ModelRegistry()
