# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for local parallel topology algebra."""

from __future__ import annotations

from vllm.config.parallel import ParallelConfig


def test_local_gpu_count_single() -> None:
    config = ParallelConfig()
    assert config.local_gpu_count == 1


def test_local_gpu_count_tp2() -> None:
    config = ParallelConfig(tensor_parallel_size=2)
    assert config.local_gpu_count == 2


def test_local_gpu_count_tp2_pp2_dp2() -> None:
    config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=2,
    )
    assert config.local_gpu_count == 8


def test_world_size_matches_tp_pp_pcp() -> None:
    config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        prefill_context_parallel_size=2,
    )
    assert config.world_size == 8


def test_dcp_must_divide_tp() -> None:
    import pytest

    with pytest.raises(ValueError, match="divisible"):
        ParallelConfig(
            tensor_parallel_size=3,
            decode_context_parallel_size=2,
        )
