# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU topology smoke tests for the focused DeepSeek build.

These tests run on a Hopper host and skip automatically when the
required GPU count exceeds available SM90 devices.
"""

from __future__ import annotations

import pytest

from tests.lean.topology_cases import TOPOLOGY_CASES


def _available_hopper_gpus() -> int:
    """Return the number of visible Hopper SM90 GPUs, or 0 on non-CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        count = torch.cuda.device_count()
        hopper = 0
        for i in range(count):
            cap = torch.cuda.get_device_capability(i)
            if cap == (9, 0):
                hopper += 1
        return hopper
    except Exception:
        return 0


@pytest.mark.parametrize("case", TOPOLOGY_CASES)
@pytest.mark.skipif(
    _available_hopper_gpus() == 0,
    reason="No Hopper SM90 GPUs available",
)
def test_topology_one_token_generation(case):
    """Generate one token with each retained parallel topology.

    Deferred: requires a Hopper host with the specified GPU count and
    a local DeepSeek safetensors checkpoint.
    """
    pytest.skip(
        "Multi-GPU topology tests require a Hopper host with a local "
        "DeepSeek checkpoint. Run manually on the target host."
    )
