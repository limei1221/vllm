# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stub. LoRA MM integration is removed from the lean build."""

from typing import Any


def set_active_mm_loras(*args: Any, **kwargs: Any) -> None:
    pass
