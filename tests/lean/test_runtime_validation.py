# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime validation tests for Hopper topology enforcement."""

from __future__ import annotations

import pytest

from vllm.config.parallel import ParallelConfig
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability


def test_rejects_non_hopper(monkeypatch) -> None:
    monkeypatch.setattr(
        CudaPlatform, "get_device_capability", lambda *_: DeviceCapability(8, 0)
    )
    with pytest.raises(RuntimeError, match="SM90 Hopper"):
        CudaPlatform.verify_hopper()


def test_accepts_hopper(monkeypatch) -> None:
    monkeypatch.setattr(
        CudaPlatform, "get_device_capability", lambda *_: DeviceCapability(9, 0)
    )
    CudaPlatform.verify_hopper()


def test_local_gpu_count_multiplies_model_and_data_ranks() -> None:
    config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        prefill_context_parallel_size=2,
        data_parallel_size=2,
    )
    assert config.local_gpu_count == 16


def test_local_gpu_count_defaults_to_one() -> None:
    config = ParallelConfig()
    assert config.local_gpu_count == 1


def test_rejects_multi_node() -> None:
    with pytest.raises(ValueError, match="single node"):
        ParallelConfig(nnodes=2)


def test_rejects_elastic_ep() -> None:
    with pytest.raises(ValueError, match="elastic expert parallelism"):
        ParallelConfig(enable_elastic_ep=True)


def test_rejects_eplb() -> None:
    with pytest.raises(ValueError, match="EPLB"):
        ParallelConfig(enable_eplb=True)
