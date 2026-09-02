# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    aiter_requires_tcp_store,
    get_distributed_init_method,
    get_file_store_init_method,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class FutureWrapper(Future):
    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response: Callable[[], Any],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        super().__init__()
        self.futures_queue.appendleft(self)

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    def _wait_for_response(self):
        try:
            response = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class MultiprocExecutor(Executor):
    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        self.monitor_workers = monitor_workers
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        set_multiprocessing_worker_envs(self.local_world_size)

        if aiter_requires_tcp_store():
            distributed_init_method = get_distributed_init_method(
                get_loopback_ip(), get_open_port()
            )
        else:
            distributed_init_method = get_file_store_init_method()
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            self.local_world_size,
            max_chunk_bytes=max_chunk_bytes,
            connect_ip=get_loopback_ip(),
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )

            for local_rank in range(self.local_world_size):
                is_driver_worker = self._is_driver_worker(local_rank)
                unready_worker_handle = WorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=local_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                )
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    assert unready_worker_handle.death_writer is not None
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            self.workers = WorkerProc.wait_for_ready(unready_workers)

            set_torch_threads_for_runtime()

            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            for rank in range(self.local_world_size):
                local_message_queue = self.workers[rank].worker_response_mq
                assert local_message_queue is not None
                self.response_mqs.append(local_message_queue)

            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque[FutureWrapper]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        return rank % self.parallel_config.tensor_parallel_size == 0

    def start_worker_monitor(self, inline=False) -> None:
        workers = self.workers
        self_ref = weakref.ref(self)

        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                return
            _self.is_failed = True
            proc = next(h.proc for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly (exit code: %s), "
                "shutting down executor.",
                proc.name,
                proc.exitcode,
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

    def sample_tokens(  # type: ignore[override]
        self, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        return self.collective_rpc(
            "sample_tokens",
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Returns single result if unique_reply_rank is provided, otherwise list."""
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called before init"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        output_rank = unique_reply_rank
        aggregate = lambda x: x

        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        future = FutureWrapper(
            self.futures_queue, get_response=get_response, aggregate=aggregate
        )

        return future if non_block else future.result()

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated."""

        def wait_for_termination(procs, timeout):
            if not time:
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
        initial_count = len(active_procs())

        logger.info(
            "[shutdown] Executor: waiting for worker exit count=%d",
            initial_count,
        )
        if wait_for_termination(
            active_procs(), timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
        ):
            logger.info_once("[shutdown] Executor: all workers exited gracefully")
            return

        remaining = active_procs()
        logger.warning(
            "[shutdown] Executor: workers still running after grace period; "
            "sending SIGTERM count=%d",
            len(remaining),
        )
        for p in remaining:
            p.terminate()
        if not wait_for_termination(active_procs(), 4):
            remaining = active_procs()
            logger.warning(
                "[shutdown] Executor: workers still running after SIGTERM; "
                "sending SIGKILL count=%d",
                len(remaining),
            )
            for p in remaining:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, "shutting_down", False):
            worker_count = len(getattr(self, "workers", None) or [])
            logger.debug(
                "[shutdown] Executor: start worker_count=%d",
                worker_count,
            )
            self.shutting_down = True

            if workers := getattr(self, "workers", None):
                for w in workers:
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                self._ensure_worker_termination([w.proc for w in workers])

                for w in workers:
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []

        logger.debug_once("[shutdown] Executor: complete")

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    def _get_output_rank(self) -> int:
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue | None
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
    ) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank
        )
        self.worker_response_mq = MessageQueue(1, 1)

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        self.worker.init_device()
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.worker.elastic_ep_execute("load_model")
        else:
            self.worker.load_model()

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        current_platform.update_block_size_for_backend(vllm_config)

        self._init_message_queues(input_shm_handle, vllm_config)

        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        proc.start()

        ready_writer.close()
        death_reader.close()
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
        )

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        e = Exception(
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        if death_pipe is None:
            return

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            try:
                death_pipe.recv()
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        Thread(
            target=death_pipe_monitor,
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()

    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker initialization and execution loops.
        This runs a background process"""

        shutdown_requested = threading.Event()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        assigned_physical_gpu_ids = kwargs[
            "vllm_config"
        ].parallel_config.assigned_physical_gpu_ids
        if assigned_physical_gpu_ids is not None:
            from vllm.platforms.interface import set_assigned_physical_gpu_ids

            set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)

        worker = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)

        try:
            worker = WorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None

            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                }
            )

            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: exiting after shutdown request"
                )
            else:
                logger.exception("WorkerProc failed.")

            shutdown_requested.set()

        except SystemExit as e:
            if shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: terminated by shutdown signal"
                )
            else:
                logger.warning("WorkerProc was terminated")
            raise e

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, output: Any):
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()
            except Exception as e:
                logger.exception("Error getting async model runner output")
                output = e

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    def handle_output(self, output: Any):
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)

    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        from vllm.platforms import current_platform

        if hasattr(self.worker, "device"):
            current_platform.set_device(self.worker.device)

        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)

                output = func(*args, **kwargs)

                if output_rank is None or self.rank == output_rank:
                    self.handle_output(output)
            except Exception as e:
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        if not model_parallel_is_initialized():
            set_process_title(name="Worker")
            decorate_logs("Worker")
            return

        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)


def set_multiprocessing_worker_envs(local_world_size: int = 1):
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()

    if current_platform.is_cpu() or "OMP_NUM_THREADS" in os.environ:
        return

    num_threads = startup_omp_num_threads(local_world_size)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ[OMP_NUM_THREADS_SET_BY_VLLM] = "1"

    torch.set_num_threads(num_threads)
    logger.debug(
        "Set OMP_NUM_THREADS=%d for %d worker process(es).",
        num_threads,
        local_world_size,
    )
