# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    ECConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class ECConnector:
    """EC connector interface used by the V2 GPU model runner."""

    @contextmanager
    def maybe_get_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> Generator[ECConnectorOutput | None, None, None]:
        yield None

    def no_forward(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT


NO_OP_EC_CONNECTOR = ECConnector()


def get_ec_connector(vllm_config: VllmConfig) -> ECConnector:
    del vllm_config
    return NO_OP_EC_CONNECTOR
