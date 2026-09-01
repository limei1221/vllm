#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused local data-parallel supervisor for single-node DeepSeek serving."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import multiprocessing
import os
import signal
from functools import partial
from http import HTTPStatus
from multiprocessing.process import BaseProcess
from typing import Iterable

import uvicorn
import uvloop
from fastapi import FastAPI, Response

import vllm.envs as envs
from vllm.entrypoints.launcher import NoSignalServer
from vllm.logger import init_logger
from vllm.utils.system_utils import (
    decorate_logs,
    kill_process_tree,
    set_process_title,
)

logger = init_logger(__name__)

CHILD_EXIT_GRACE_S = 5.0


class LocalDPRouter:
    """Round-robin router across local DP engine endpoints.

    Assigns each request to the next healthy local engine in order.
    Fails fast when all engines are dead.
    """

    def __init__(self, engine_names: Iterable[str]):
        self._engines: list[str] = list(engine_names)
        self._healthy: set[str] = set(self._engines)
        self._index: int = 0

    def next_engine(self) -> str:
        """Return the next healthy engine name.

        Raises:
            RuntimeError: If no local DP engine is healthy.
        """
        if not self._healthy:
            raise RuntimeError("All local DP engines have failed.")
        n = len(self._engines)
        for _ in range(n):
            engine = self._engines[self._index % n]
            self._index += 1
            if engine in self._healthy:
                return engine
        raise RuntimeError("local DP engine failed")

    def mark_failed(self, engine: str) -> None:
        """Mark an engine as failed."""
        self._healthy.discard(engine)


class DPSupervisor:
    """Launches and monitors local DP engine subprocesses."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.supervisor_port = getattr(args, "data_parallel_supervisor_port", 0) or 0
        self.child_ports = [
            args.port + local_rank
            for local_rank in range(args.data_parallel_size_local)
        ]
        self._is_ready = False
        self._processes: list[BaseProcess] = []
        self._shutdown_event = asyncio.Event()
        self._shutdown_signal = signal.SIGTERM
        self._router: LocalDPRouter | None = None

    @property
    def is_ready(self) -> bool:
        return self._is_ready and not self._shutdown_event.is_set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        decorate_logs("DPSupervisor")

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, partial(self._handle_signal, sig))

        supervisor_server: uvicorn.Server | None = None
        supervisor_server_task: asyncio.Task[None] | None = None
        try:
            self._start_children()
            monitor_task = asyncio.create_task(
                self._monitor_children(), name="dp-monitor"
            )

            await self._wait_until_ready(monitor_task)
            if self.is_ready and not monitor_task.done():
                supervisor_server, supervisor_server_task = await self._start_server()

            await monitor_task
        finally:
            self._shutdown_children()
            if supervisor_server is not None and not supervisor_server.should_exit:
                supervisor_server.should_exit = True
            if supervisor_server_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await supervisor_server_task

    def _handle_signal(self, sig: int) -> None:
        logger.info("Received signal %s, shutting down.", sig)
        self._shutdown_signal = sig
        self._shutdown_event.set()
        self._shutdown_children()

    def _start_children(self) -> None:
        """Start one local API server process per local DP rank."""
        ctx = multiprocessing.get_context("spawn")
        for local_rank in range(self.args.data_parallel_size_local):
            child_args = _make_child_args(self.args, local_rank)
            proc = ctx.Process(
                target=_run_child,
                args=(child_args,),
                name=f"VllmDPServer-{local_rank}",
                daemon=True,
            )
            proc.start()
            self._processes.append(proc)
            logger.info(
                "Started DP server rank %d (pid=%d, port=%d)",
                local_rank,
                proc.pid,
                child_args.port,
            )

        self._router = LocalDPRouter(
            f"localhost:{port}" for port in self.child_ports
        )

    async def _monitor_children(self) -> None:
        """Monitor child process health; set shutdown event if any dies."""
        while not self._shutdown_event.is_set():
            for proc in self._processes:
                if not proc.is_alive() and not self._shutdown_event.is_set():
                    logger.error(
                        "DP server pid=%d exited unexpectedly (code=%s)",
                        proc.pid,
                        proc.exitcode,
                    )
                    self._shutdown_event.set()
                    return
            await asyncio.sleep(1.0)

    async def _wait_until_ready(self, monitor_task: asyncio.Task) -> None:
        """Wait until all child servers respond to /health or a child dies."""
        import aiohttp

        while not self._shutdown_event.is_set():
            if monitor_task.done():
                return
            all_ready = True
            for port in self.child_ports:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://localhost:{port}/health", timeout=aiohttp.ClientTimeout(total=2)
                        ) as resp:
                            if resp.status != 200:
                                all_ready = False
                except Exception:
                    all_ready = False
                    break
            if all_ready:
                self._is_ready = True
                logger.info("All DP servers are ready.")
                return
            await asyncio.sleep(1.0)

    async def _start_server(self):
        """Start the supervisor reverse-proxy server."""
        app = FastAPI()
        app.state.router = self._router
        app.state.child_ports = self.child_ports

        @app.get("/health")
        async def health():
            if not self.is_ready:
                return Response(status_code=HTTPStatus.SERVICE_UNAVAILABLE)
            return Response(status_code=HTTPStatus.OK)

        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def proxy(path: str, request: dict):
            assert self._router is not None
            engine = self._router.next_engine()
            port = self._router._engines[
                self._router._index % len(self._router._engines)
            ].split(":")[1]
            # Forward to the selected engine
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.get("method", "GET"),
                    url=f"http://localhost:{port}/{path}",
                    json=request.get("body"),
                ) as resp:
                    return Response(
                        content=await resp.read(),
                        status_code=resp.status,
                        headers=dict(resp.headers),
                    )

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.args.port,
            log_level=self.args.log_level,
        )
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        return server, server_task

    def _shutdown_children(self) -> None:
        """Terminate all child processes."""
        for proc in self._processes:
            if proc.is_alive():
                kill_process_tree(proc.pid)
        self._processes.clear()


def _make_child_args(args: argparse.Namespace, local_rank: int) -> argparse.Namespace:
    """Create args for a child DP server process."""
    child = argparse.Namespace(**vars(args))
    child.port = args.port + local_rank
    child.data_parallel_rank = (
        getattr(args, "data_parallel_rank", 0) or 0
    ) + local_rank
    return child


def _run_child(args: argparse.Namespace) -> None:
    """Run a single DP server child process."""
    decorate_logs(f"DPServer-rank{args.data_parallel_rank}")
    set_process_title(f"VllmDPServer-{args.data_parallel_rank}")

    from vllm.entrypoints.openai.api_server import run_server

    os.environ["VLLM_DP_RANK"] = str(args.data_parallel_rank)
    run_server(args)
