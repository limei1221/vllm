# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import Request

from vllm import PromptType, SamplingParams, envs
from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import (
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from vllm.entrypoints.serve import create_error_response
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import EngineInput
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from vllm.sampling_params import BeamSearchParams
from vllm.utils import random_uuid


class BaseServing:
    def __init__(
        self,
        models: OpenAIServingModels | OpenAIModelRegistry,
        model_config: ModelConfig,
        request_logger: RequestLogger | None = None,
    ):
        self.models = models
        self.model_config = model_config
        self.request_logger = request_logger

    async def _check_model(
        self,
        request: AnyRequest,
    ) -> ErrorResponse | None:
        error_response = None

        if self._is_model_supported(request.model):
            return None

        return error_response or self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        if envs.VLLM_SKIP_MODEL_NAME_VALIDATION:
            return True
        return self.models.is_base_model(model_name)

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    def _extract_prompt_components(self, prompt: PromptType | EngineInput):
        return extract_prompt_components(self.model_config, prompt)

    def _extract_prompt_text(self, prompt: PromptType | EngineInput):
        return self._extract_prompt_components(prompt).text

    def _extract_prompt_len(self, prompt: EngineInput):
        return extract_prompt_len(self.model_config, prompt)

    def _log_inputs(
        self,
        request_id: str,
        inputs: PromptType | EngineInput,
        params: SamplingParams | BeamSearchParams | None,
    ) -> None:
        if self.request_logger is None:
            return

        components = self._extract_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            components.text,
            components.token_ids,
            components.embeds,
            params=params,
        )

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str | None:
        """Pulls the request id to use from a header, if provided"""
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default

    def _get_message_types(self, request: AnyRequest) -> set[str]:
        """Retrieve the set of types from message content dicts up
        until `_`; we use this to match potential extra data
        with default per modality loras.
        """
        message_types: set[str] = set()

        if not hasattr(request, "messages"):
            return message_types

        messages = request.messages
        if messages is None or isinstance(messages, (str, bytes)):
            return message_types

        for message in messages:
            if (
                isinstance(message, dict)
                and "content" in message
                and isinstance(message["content"], list)
            ):
                for content_dict in message["content"]:
                    if "type" in content_dict:
                        message_types.add(content_dict["type"].split("_")[0])
        return message_types
