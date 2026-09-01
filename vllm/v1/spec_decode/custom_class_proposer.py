# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stub. Custom class proposer is removed from the lean build."""

from typing import Any


def create_custom_proposer(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("custom_class proposer is not supported in the lean build")
