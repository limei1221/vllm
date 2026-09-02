# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.logger import init_logger

router = APIRouter()

logger = init_logger(__name__)


def base(request: Request) -> ServingTokenization:
    # Reuse the existing instance
    return tokenization(request)


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client
