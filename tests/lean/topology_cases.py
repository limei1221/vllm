# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finite topology matrix for multi-GPU smoke tests."""

import pytest

TOPOLOGY_CASES = (
    pytest.param(dict(tp=2), id="tp2"),
    pytest.param(dict(pp=2), id="pp2"),
    pytest.param(dict(dp=2), id="dp2"),
    pytest.param(dict(tp=2, ep=True), id="tp2-ep"),
    pytest.param(dict(tp=2, dcp=2), id="tp2-dcp2"),
    pytest.param(dict(pcp=2), id="pcp2"),
    pytest.param(dict(tp=2, pp=2, dp=2), id="tp2-pp2-dp2"),
)
