# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.parser.abstract_parser import Parser

logger = init_logger(__name__)


class ParserManager:
    """Resolves the Parser used by the chat path.

    Reasoning and tool parsers are not part of this build, so no parser is
    ever resolved and callers fall back to returning raw model output.
    """

    @classmethod
    def get_parser(
        cls,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
        is_harmony: bool = False,
    ) -> type[Parser] | None:
        """Return None; this build has no reasoning or tool parsers."""
        if tool_parser_name or reasoning_parser_name or enable_auto_tools:
            logger.warning_once(
                "Reasoning and tool parsers are not supported by this build; "
                "model output is returned unparsed."
            )
        return None
