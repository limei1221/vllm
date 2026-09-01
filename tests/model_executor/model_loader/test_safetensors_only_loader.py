# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the safetensors-only model loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.safetensors_loader import (
    ResolvedSafetensors,
    SafetensorsModelLoader,
)


@pytest.fixture
def safetensors_snapshot(tmp_path: Path) -> Path:
    (tmp_path / "model.safetensors").touch()
    return tmp_path


def test_load_config_accepts_only_safetensors() -> None:
    assert LoadConfig().load_format == "safetensors"
    with pytest.raises(ValueError, match="only safetensors"):
        LoadConfig(load_format="pt")


def test_missing_safetensors_is_explicit(tmp_path: Path) -> None:
    loader = SafetensorsModelLoader(LoadConfig())
    with pytest.raises(RuntimeError, match="No safetensors weights"):
        loader.prepare_weights(str(tmp_path), revision=None)


def test_hub_id_resolves_snapshot(monkeypatch, safetensors_snapshot: Path) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.safetensors_loader.download_weights_from_hf",
        lambda model, *a, **kw: str(safetensors_snapshot),
    )
    loader = SafetensorsModelLoader(LoadConfig())
    resolved = loader.prepare_weights("org/deepseek-test", revision="main")
    assert resolved.directory == safetensors_snapshot
    assert all(path.suffix == ".safetensors" for path in resolved.files)


def test_local_directory_resolves_safetensors(
    safetensors_snapshot: Path,
) -> None:
    loader = SafetensorsModelLoader(LoadConfig())
    resolved = loader.prepare_weights(str(safetensors_snapshot), revision=None)
    assert resolved.directory == safetensors_snapshot
    assert len(resolved.files) == 1
    assert resolved.files[0].suffix == ".safetensors"
