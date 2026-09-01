# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stub. Punica wrapper is removed from the lean build."""

from typing import Any


class PunicaWrapperBase:
    """No-op stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def get_punica_wrapper(*args: Any, **kwargs: Any) -> Any:
    return None
