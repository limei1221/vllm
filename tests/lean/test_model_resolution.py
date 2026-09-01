# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for direct DeepSeek model resolution."""

from __future__ import annotations

import pytest

from vllm.model_executor.models.registry import (
    SUPPORTED_ARCHITECTURES,
    resolve_deepseek_model_class,
)


@pytest.mark.parametrize("architecture", list(SUPPORTED_ARCHITECTURES))
def test_resolves_supported_architecture(architecture: str) -> None:
    assert resolve_deepseek_model_class(architecture).__name__


def test_rejects_other_architecture() -> None:
    with pytest.raises(ValueError, match="DeepSeek V2/V3"):
        resolve_deepseek_model_class("LlamaForCausalLM")


def test_registry_has_exactly_six_architectures() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    assert set(ModelRegistry.get_supported_archs()) == {
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepSeekMTPModel",
        "EagleDeepSeekMTPModel",
        "Eagle3DeepseekV2ForCausalLM",
        "Eagle3DeepseekV3ForCausalLM",
    }
