# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `vllm serve` subcommand."""

import argparse

import uvloop

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

DESCRIPTION = """Launch a local OpenAI-compatible API server serving DeepSeek
V2/V3 completions over HTTP.

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.
--help=ModelConfig, --help=Frontend)
  Use `--help=all` to show all available flags at once.
"""


def cmd(args: argparse.Namespace) -> None:
    # The model may be given as a positional argument, which takes precedence
    # over `--model`.
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag

    if args.grpc:
        from vllm.entrypoints.grpc_server import serve_grpc

        uvloop.run(serve_grpc(args))
        return

    if args.headless:
        raise NotImplementedError(
            "Headless mode is not supported by this build; it exists to serve "
            "remote data-parallel ranks, and only local data parallelism is "
            "supported."
        )

    # More than one local DP rank needs a supervisor to fan requests out across
    # one API server per rank.
    if (args.data_parallel_size_local or 0) > 1:
        from vllm.entrypoints.openai.dp_supervisor import run_dp_supervisor

        setup_multiprocess_prometheus()
        run_dp_supervisor(args)
        return

    from vllm.entrypoints.openai.api_server import run_server

    uvloop.run(run_server(args))


def subparser_init(
    subparsers: argparse._SubParsersAction,
) -> FlexibleArgumentParser:
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the vLLM OpenAI-compatible API server.",
        description=DESCRIPTION,
        usage="vllm serve [model_tag] [options]",
    )
    serve_parser = make_arg_parser(serve_parser)
    serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="serve")
    return serve_parser
