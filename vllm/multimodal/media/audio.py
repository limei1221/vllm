# SPDX-License-Identifier: Apache-2.0
from typing import Any


def load_audio(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("multimodal audio support has been removed")


def load_audio_pyav(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("multimodal audio support has been removed")
