# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs. Elastic EP is removed from the lean build."""

from typing import Any


class ElasticEPScalingExecutor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return None


class ElasticEPScalingState:
    commit_requested: bool = False
    worker_type: Any = None
    ready_key: Any = None
    run_pre_kv_init_states: Any = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def is_ready_for_switch(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def is_complete(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def progress(self, *args: Any, **kwargs: Any) -> None:
        pass


def get_standby_ep_group(*args: Any, **kwargs: Any) -> Any:
    return None
