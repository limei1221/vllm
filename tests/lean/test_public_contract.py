# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public contract characterization for the lean DeepSeek vertical slice.

These tests pin the retained public roots and the supported architecture
names so that later reduction tasks cannot silently regress the surface.
"""

from __future__ import annotations

from types import SimpleNamespace


def test_retained_public_roots_import() -> None:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.api_server import build_app
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    assert LLM is not None
    assert SamplingParams is not None
    assert build_app is not None
    assert MultiprocExecutor is not None


def test_executor_class_uses_uni_for_one_rank() -> None:
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=1))
    assert Executor.get_class(config) is UniProcExecutor


def test_executor_class_uses_local_mp() -> None:
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))
    assert Executor.get_class(config) is MultiprocExecutor


def test_only_deepseek_architectures_are_registered() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    assert set(ModelRegistry.get_supported_archs()) == {
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepSeekMTPModel",
        "EagleDeepSeekMTPModel",
        "Eagle3DeepseekV2ForCausalLM",
        "Eagle3DeepseekV3ForCausalLM",
    }
