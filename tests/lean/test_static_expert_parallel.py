# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for static expert parallel placement."""

from __future__ import annotations

from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)


def test_linear_expert_placement() -> None:
    assert compute_local_expert_ids(num_experts=8, ep_size=2, ep_rank=0) == {0, 1, 2, 3}
    assert compute_local_expert_ids(num_experts=8, ep_size=2, ep_rank=1) == {4, 5, 6, 7}


def test_linear_expert_placement_single_rank() -> None:
    assert compute_local_expert_ids(num_experts=4, ep_size=1, ep_rank=0) is None


def test_linear_expert_placement_all_experts() -> None:
    result = compute_local_expert_ids(num_experts=4, ep_size=4, ep_rank=0)
    assert result == {0}
    result = compute_local_expert_ids(num_experts=4, ep_size=4, ep_rank=3)
    assert result == {3}
