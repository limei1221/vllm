# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from typing import Literal, TypeVar, overload

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


class Executor(ABC):
    """Abstract base class for vLLM executors.

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on
    multiple devices.
    """

    uses_ray: bool = False
    supports_pp: bool = False  # whether the executor supports PP

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        parallel_config = vllm_config.parallel_config
        if parallel_config.world_size == 1:
            from vllm.v1.executor.uniproc_executor import UniProcExecutor

            return UniProcExecutor
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        return MultiprocExecutor

    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Initialize the KV caches on the underlying workers."""
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))

    def compile_or_warm_up_model(self) -> None:
        """Compile/warm up the model and capture cudagraphs on workers."""
        compilation_times: list[CompilationTimes] = self.collective_rpc(
            "compile_or_warm_up_model"
        )
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                t.encoder for t in compilation_times
            )

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return self.collective_rpc("get_kv_cache_spec")

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]: ...

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]: ...

    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        raise NotImplementedError

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None: ...

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]: ...

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput: ...

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]: ...

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.collective_rpc("shutdown")

    @cached_property  # Avoid unnecessary RPC calls
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output: list[tuple[SupportedTask, ...]]
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def supports_draft_weight_updates(self) -> bool:
        worker_support: list[bool] = self.collective_rpc(
            "supports_draft_weight_updates"
        )
        return all(worker_support)

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """Whether the executor supports async scheduling."""
        return False
