# SPDX-License-Identifier: Apache-2.0
"""No-op stubs. Elastic EP is removed from the lean build."""

from typing import Any


class ElasticEPScalingExecutor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ElasticEPScalingState:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def get_standby_ep_group(*args: Any, **kwargs: Any) -> Any:
    return None
