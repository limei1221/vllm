# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stub. Multimodal pruning is removed from the lean build."""

from typing import Any


def maybe_create_mm_pruner(*args: Any, **kwargs: Any):
    return None
