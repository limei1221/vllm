# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import cast

import msgspec
import zmq

from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    get_open_port,
    get_open_zmq_ipc_path,
    get_tcp_uri,
    zmq_socket_ctx,
)
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.executor import Executor
from vllm.v1.utils import _SubprocessWrapper, get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)

# Environment variables that must not be inherited by worker processes.
WORKER_SPECIFIC_ENV_VARS: set[str] = {
    "VLLM_HOST_IP",
    "VLLM_HOST_PORT",
    "LOCAL_RANK",
    "CUDA_VISIBLE_DEVICES",
}


STARTUP_POLL_PERIOD_MS = 10000


class CoreEngineState(Enum):
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


@dataclass
class EngineZmqAddresses:
    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: str | None = None
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: str | None = None
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: str | None = None


@dataclass
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """

    addresses: EngineZmqAddresses
    parallel_config: dict[str, int | str | list[int]]


class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        context = get_mp_context()
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }

        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        is_dp = vllm_config.parallel_config.data_parallel_size > 1

        from vllm.v1.engine.core import EngineCoreProc

        self.processes: list[BaseProcess] = []
        local_dp_ranks = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            # Start EngineCore in background process.
            local_dp_ranks.append(local_index)
            self.processes.append(
                context.Process(
                    target=EngineCoreProc.run_engine_core,
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    kwargs=common_kwargs
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                )
            )

        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        # All ranks share this config object: capture the user-provided
        # --device-ids list before the per-rank shard overwrites it. Mutating
        # the config before each proc.start() works because the spawn method
        # pickles process args at start() time, sequentially per rank.
        user_assigned_gpu_ids = vllm_config.parallel_config.assigned_physical_gpu_ids
        try:
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                # Populate the logical-to-physical GPU mapping in DP for
                # platforms that cannot rely on
                # torch.accelerator.set_device_index().
                needs_device_env_isolation = not (
                    current_platform.is_cuda_alike() or current_platform.is_xpu()
                )
                if is_dp and needs_device_env_isolation:
                    set_assigned_physical_gpu_ids_for_dp_rank(
                        vllm_config, local_dp_rank, user_assigned_gpu_ids
                    )

                proc.start()
        finally:
            # Kill other procs if not all are running.
            if self.finished_procs():
                self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine core processes with configurable timeout."""
        self.manager_stopped.set()
        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)

    def monitor_engine_liveness(self) -> None:
        """Monitor engine core process liveness."""

        sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
        sentinels = set(sentinel_to_proc.keys())

        while sentinels and not self.manager_stopped.is_set():
            died_sentinels = connection.wait(sentinels, timeout=1)

            for sentinel in died_sentinels:
                proc = sentinel_to_proc.pop(cast(int, sentinel))
                exitcode = proc.exitcode
                if exitcode != 0 and not self.manager_stopped.is_set():
                    self.failed_proc_name = proc.name
            if died_sentinels:
                break

        self.shutdown()

    def sentinels(self) -> list:
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


class SignalCallback:
    """Safely trigger a callback from signal handler context via a dedicated thread."""

    def __init__(self, callback: Callable[[], None]):
        self._callback = callback
        self._event = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="signal-callback",
        )
        self._thread.start()

    def _run(self):
        self._event.wait()
        if not self._stopped:
            self._callback()

    def trigger(self):
        self._event.set()

    def stop(self):
        self._stopped = True
        self._event.set()


def set_assigned_physical_gpu_ids_for_dp_rank(
    vllm_config: VllmConfig,
    local_dp_rank: int,
    user_assigned_gpu_ids: list[int] | None = None,
) -> None:
    """
    Populate assigned_physical_gpu_ids on the config for the given DP rank.

    user_assigned_gpu_ids is the full (un-sharded) --device-ids list, if the
    user provided one; this DP rank's shard is sliced from it. It is passed
    explicitly rather than read from the config because callers may reuse
    one config object across DP ranks, overwriting the field each time.
    """
    world_size = vllm_config.parallel_config.world_size
    local_world_size = vllm_config.parallel_config.local_world_size
    evar = current_platform.device_control_env_var

    physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
        evar,
        local_dp_rank,
        world_size,
        local_world_size,
        user_assigned_gpu_ids=user_assigned_gpu_ids,
    )
    vllm_config.parallel_config.assigned_physical_gpu_ids = physical_gpu_ids


def get_physical_gpu_ids_for_local_dp_rank(
    device_control_env_var: str,
    local_dp_rank: int,
    world_size: int,
    local_world_size: int | None = None,
    user_assigned_gpu_ids: list[int] | None = None,
) -> list[int]:
    """
    Returns list of physical GPU IDs for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will return [2, 3] for local_dp_rank=1.

    If user_assigned_gpu_ids is provided (e.g. from --device-ids), this DP
    rank's shard is sliced from it instead of being derived from the
    device-control env var.
    """
    if local_world_size is None:
        local_world_size = world_size
    if user_assigned_gpu_ids is not None:
        start = local_dp_rank * world_size
        stop = start + local_world_size
        if stop > len(user_assigned_gpu_ids):
            raise ValueError(
                f"--device-ids provides {len(user_assigned_gpu_ids)} devices, "
                f"but DP rank {local_dp_rank} needs devices [{start}, {stop})"
            )
        return user_assigned_gpu_ids[start:stop]
    try:
        return [
            current_platform.device_id_to_physical_device_id(i)
            for i in range(
                local_dp_rank * world_size,
                local_dp_rank * world_size + local_world_size,
            )
        ]
    except IndexError as e:
        raise Exception(
            f"Error computing device indices for "
            f"{device_control_env_var}: "
            f"local range: [{local_dp_rank * world_size}, "
            f"{(local_dp_rank + 1) * world_size}) "
            "base value: "
            f'"{os.getenv(device_control_env_var)}"'
        ) from e


def _apply_dp_identity_suffix(dp_vllm_config, dp_rank: int) -> None:
    # KV-connector engine_ids must
    # be unique across sibling DP engines or registration collides.
    # Use the global DP rank, not a node-local rank, since sibling DP
    # engines can span multiple nodes.
    dp_vllm_config.instance_id = f"{dp_vllm_config.instance_id}_dp{dp_rank}"
    if dp_vllm_config.kv_transfer_config is not None:
        dp_vllm_config.kv_transfer_config.engine_id = (
            f"{dp_vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
        )


def get_engine_zmq_addresses(
    vllm_config: VllmConfig,
    num_api_servers: int = 1,
    *,
    defer_api_server_ports: bool = True,
) -> EngineZmqAddresses:
    """Allocate ZMQ addresses for engine-client communication.

    By default each TCP address is a ``tcp://host:0`` placeholder; the
    consumer (API-server child or single-process ``MPClient``) binds, then
    recovers the kernel-assigned port via ``getsockopt(zmq.LAST_ENDPOINT)``
    and writes it back into ``addresses`` before the engine handshake.

    Set ``defer_api_server_ports=False`` only when the consumer cannot
    report a bound port back (e.g. the Rust front-end). IPC paths are
    unaffected."""
    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_size = parallel_config.data_parallel_size
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    # In offline mode there is an LLM instance per DP rank and
    # one core engine per LLM, see
    # examples/features/data_parallel/data_parallel_offline.py.
    offline_mode = local_start_index is not None

    # client_local_only = True for cases where this front-end
    # sends requests only to colocated engines.
    client_local_only = (
        offline_mode or local_engines_only or (local_engine_count == dp_size)
    )
    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        client_local_only = False

    def _addr() -> str:
        if client_local_only:
            return get_open_zmq_ipc_path()
        return get_tcp_uri(host, 0 if defer_api_server_ports else get_open_port())

    return EngineZmqAddresses(
        inputs=[_addr() for _ in range(num_api_servers)],
        outputs=[_addr() for _ in range(num_api_servers)],
    )


FrontendProcess = BaseProcess | _SubprocessWrapper


@dataclass
class CoreEngineLaunch:
    """Resources and startup barrier for launched engine processes."""

    engine_manager: CoreEngineProcManager | None
    coordinator: DPCoordinator | None
    addresses: EngineZmqAddresses
    tensor_queue: Queue | None
    # Frontend processes to watch during engine startup; may be assigned by
    # the caller before the startup barrier runs on context manager exit.
    watched_frontend_processes: Sequence[FrontendProcess] = ()


@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
) -> Iterator[CoreEngineLaunch]:
    """Launch engine and DP coordinator processes as needed."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    tensor_queue: Queue | None = None

    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # The coordinator is needed for:
    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
    # 2. MoE models: wave coordination in addition to stats
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
    else:
        coordinator = None

    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
    else:
        # Rank > 0 handshakes with just the local cores it is managing.
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    handshake_local_only = offline_mode or local_engine_count == dp_size

    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    handshake_address = get_engine_client_zmq_addr(
        handshake_local_only,
        host,
        parallel_config.data_parallel_rpc_port,
    )

    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:
        # Start local engines.
        if local_engine_count:
            local_engine_manager = CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        launch = CoreEngineLaunch(
            local_engine_manager, coordinator, addresses, tensor_queue
        )
        yield launch
        wait_for_engine_startup(
            handshake_socket,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            launch,
        )


def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    coordinated_dp: bool,
    cache_config: CacheConfig,
    launch: CoreEngineLaunch,
):
    # Wait for engine core process(es) to send ready messages.
    local_count = parallel_config.data_parallel_size_local
    remote_count = len(core_engines) - local_count
    # [local, remote] counts
    conn_pending, start_pending = [local_count, remote_count], [0, 0]
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    remote_should_be_headless = (
        not parallel_config.data_parallel_hybrid_lb
        and not parallel_config.data_parallel_external_lb
    )

    # 1. Engine processes
    if isinstance(launch.engine_manager, CoreEngineProcManager):
        for sentinel in launch.engine_manager.sentinels():
            poller.register(sentinel, zmq.POLLIN)
    # 2. DP Coordinator process, if present
    coord_process = launch.coordinator.proc if launch.coordinator else None
    if coord_process is not None:
        poller.register(coord_process.sentinel, zmq.POLLIN)
    # 3. Watched frontend processes, if any
    frontend_process_by_fd: dict[int, FrontendProcess] = {}
    for proc in launch.watched_frontend_processes:
        fd = proc.sentinel if isinstance(proc.sentinel, int) else proc.sentinel.fileno()
        frontend_process_by_fd[fd] = proc
        poller.register(fd, zmq.POLLIN)

    while any(conn_pending) or any(start_pending):
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if not events:
            if any(conn_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,
                )
            if any(start_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,
                )
            continue
        if len(events) > 1 or events[0][0] != handshake_socket:
            # One of the local core, coordinator, or watched frontend processes exited.
            if isinstance(launch.engine_manager, CoreEngineProcManager):
                finished = launch.engine_manager.finished_procs()
            else:
                finished = {}
            if coord_process is not None and coord_process.exitcode is not None:
                finished[coord_process.name] = coord_process.exitcode
            failed_frontend_procs = {
                proc.name: proc.exitcode
                for fd, proc in frontend_process_by_fd.items()
                if proc.exitcode is not None
                or any(event_fd == fd for event_fd, _ in events)
            }
            if failed_frontend_procs and not finished:
                raise RuntimeError(
                    "Frontend process failed during engine core initialization. "
                    "See root cause above. "
                    f"Failed frontend proc(s): {failed_frontend_procs}"
                )
            raise RuntimeError(
                "Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {finished}"
                + (
                    f", failed frontend proc(s): {failed_frontend_procs}"
                    if failed_frontend_procs
                    else ""
                )
            )

        # Receive HELLO and READY messages from the input socket.
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()
        eng_index = int.from_bytes(eng_identity, "little")
        engine = next((e for e in core_engines if e.identity == eng_identity), None)
        if engine is None:
            raise RuntimeError(
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )
        msg = msgspec.msgpack.decode(ready_msg_bytes)
        status, local, headless = msg["status"], msg["local"], msg["headless"]
        if local != engine.local:
            raise RuntimeError(
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # Remote engines must be headless iff we aren't in hybrid dp lb mode.
        if not local and headless != remote_should_be_headless:
            if headless:
                raise RuntimeError(
                    f"Remote engine {eng_index} must not use "
                    f"--headless in external or hybrid dp lb "
                    f"mode"
                )
            else:
                raise RuntimeError(
                    f"Remote engine {eng_index} must use "
                    f"--headless unless in external or hybrid "
                    f"dp lb mode"
                )

        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            # Send init message with DP config info.
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=launch.addresses,
                    parallel_config={
                        k: getattr(parallel_config, k)
                        for k in (
                            "data_parallel_master_ip",
                            "data_parallel_master_port",
                            "_data_parallel_master_port_list",
                            "data_parallel_size",
                        )
                    }
                    if coordinated_dp
                    else {},
                )
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            conn_pending[0 if local else 1] -= 1
            start_pending[0 if local else 1] += 1
            engine.state = CoreEngineState.CONNECTED
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # Validate config hash consistency across DP workers for MoE models.
            if coordinated_dp:
                worker_config_hash = msg.get("parallel_config_hash")
                expected_hash = parallel_config.compute_hash()
                if worker_config_hash != expected_hash:
                    raise RuntimeError(
                        f"Configuration mismatch detected for engine "
                        f"{eng_index}. All DP workers must have identical "
                        f"configurations for parameters that affect collective "
                        f"communication (e.g., enable_eplb, "
                        f"eplb_config.log_balancedness). "
                        f"Worker hash: {worker_config_hash}, "
                        f"Expected hash: {expected_hash}. "
                        f"Please ensure all workers are started with the same "
                        f"command-line arguments."
                    )

            start_pending[0 if local else 1] -= 1
            engine.state = CoreEngineState.READY
        else:
            raise RuntimeError(
                f"Unexpected {status} message for "
                f"{'local' if local else 'remote'} engine "
                f"{eng_index} in {engine.state} state."
            )

        logger.debug(
            "%s from %s core engine process %s.",
            status,
            "local" if local else "remote",
            eng_index,
        )
