# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the focused offline LLM API surface."""

from __future__ import annotations


def test_llm_exposes_only_retained_inference_methods() -> None:
    from vllm import LLM

    assert hasattr(LLM, "generate")
    assert hasattr(LLM, "chat")
    assert not hasattr(LLM, "beam_search")
    assert not hasattr(LLM, "encode")
    assert not hasattr(LLM, "score")
