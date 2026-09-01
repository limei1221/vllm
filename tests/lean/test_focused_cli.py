# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the focused CLI and configuration surface."""

from __future__ import annotations

import pytest


def test_cli_rejects_lora() -> None:
    from vllm.engine.arg_utils import EngineArgs

    with pytest.raises((ValueError, SystemExit, TypeError)):
        EngineArgs(model="test", enable_lora=True)


def test_speculative_config_accepts_mtp() -> None:
    from vllm.config.speculative import SpeculativeConfig

    config = SpeculativeConfig(method="mtp", num_speculative_tokens=1)
    assert config.method == "mtp"


def test_speculative_config_accepts_eagle() -> None:
    from vllm.config.speculative import SpeculativeConfig

    config = SpeculativeConfig(method="eagle", num_speculative_tokens=1)
    assert config.method == "eagle"


def test_speculative_config_accepts_eagle3() -> None:
    from vllm.config.speculative import SpeculativeConfig

    config = SpeculativeConfig(method="eagle3", num_speculative_tokens=1)
    assert config.method == "eagle3"


@pytest.mark.parametrize("method", ["ngram", "suffix", "medusa", "draft_model"])
def test_rejects_generic_speculative_methods(method: str) -> None:
    from vllm.config.speculative import SpeculativeConfig

    with pytest.raises(ValueError, match="MTP or EAGLE"):
        SpeculativeConfig(method=method, num_speculative_tokens=1)
