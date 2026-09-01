# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op stubs. Tool parser utils are removed from the lean build."""

from typing import Any


def build_responses_tool_call_name_map(*args: Any, **kwargs: Any) -> Any:
    return {}


def flat_namespace_tool_name(*args: Any, **kwargs: Any) -> str:
    return ""


def iter_response_function_tool_dicts(*args: Any, **kwargs: Any) -> Any:
    return iter([])


def resolve_responses_tool_call_name(*args: Any, **kwargs: Any) -> str:
    return ""
