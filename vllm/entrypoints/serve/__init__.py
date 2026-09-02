# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI

from vllm.logger import init_logger

from .exception_handling.error_response import create_error_response

logger = init_logger(__name__)

__all__ = [
    "create_error_response",
    "register_vllm_serve_api_routers",
]


def register_vllm_serve_api_routers(app: FastAPI):
    from .instrumentator import register_instrumentator_api_routers

    register_instrumentator_api_routers(app)

    from vllm.entrypoints.serve.profile.api_router import (
        attach_router as attach_profile_router,
    )

    attach_profile_router(app)

    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)
