# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoint of vLLM.

Modules are loaded lazily within `main` to avoid eager import breakage.
"""

import importlib.metadata


def main():
    import vllm.entrypoints.cli.serve
    from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args
    from vllm.entrypoints.serve.utils.api_utils import (
        VLLM_SUBCMD_PARSER_EPILOG,
        cli_env_setup,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    vllm.entrypoints.cli.serve.subparser_init(subparsers)

    args = parser.parse_args()
    if args.subparser != "serve":
        parser.print_help()
        return

    validate_parsed_serve_args(args)
    vllm.entrypoints.cli.serve.cmd(args)


if __name__ == "__main__":
    main()
