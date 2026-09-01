# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LocalDPRouter round-robin and failure behavior."""

from __future__ import annotations

import pytest

from vllm.entrypoints.openai.dp_supervisor import LocalDPRouter


def test_local_dp_router_round_robins_healthy_engines() -> None:
    router = LocalDPRouter(["engine-0", "engine-1"])
    assert [router.next_engine() for _ in range(4)] == [
        "engine-0",
        "engine-1",
        "engine-0",
        "engine-1",
    ]


def test_local_dp_router_fails_when_engine_dies() -> None:
    router = LocalDPRouter(["engine-0"])
    router.mark_failed("engine-0")
    with pytest.raises(RuntimeError, match="local DP engine failed"):
        router.next_engine()


def test_local_dp_router_skips_failed_engines() -> None:
    router = LocalDPRouter(["engine-0", "engine-1", "engine-2"])
    router.mark_failed("engine-1")
    results = [router.next_engine() for _ in range(4)]
    assert "engine-1" not in results
    assert results == ["engine-0", "engine-2", "engine-0", "engine-2"]
