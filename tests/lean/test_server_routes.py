# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the focused five-route server surface."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest


def _make_args() -> Namespace:
    return Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=True,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        middleware=[],
        enable_fault_tolerance=False,
        enable_lora=False,
    )


def test_server_has_only_supported_routes() -> None:
    from vllm.entrypoints.openai.api_server import build_app

    app = build_app(_make_args())
    paths = {
        route.path
        for route in app.routes
        if hasattr(route, "path") and route.path != "/openapi.json"
    }
    assert paths == {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/models",
        "/health",
        "/metrics",
    }
