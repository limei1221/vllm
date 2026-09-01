# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stub. Multimodal encoder runner is removed from the lean build."""

from contextlib import contextmanager
from typing import Any

import torch


class EncoderRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.inputs_embeds = torch.empty(0)

    def get_inputs_embeds(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.inputs_embeds

    def prepare_mm_inputs(self, *args: Any, **kwargs: Any) -> tuple:
        return [], {}

    def execute_mm_encoder(self, *args: Any, **kwargs: Any) -> list:
        return []

    def gather_mm_embeddings(self, *args: Any, **kwargs: Any) -> tuple:
        return [], torch.empty(0, dtype=torch.bool)

    @contextmanager
    def timed_encoder_operation(self, *args: Any, **kwargs: Any):
        yield
