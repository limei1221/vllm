# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, overload

from typing_extensions import assert_never

from vllm.config import VllmConfig
from vllm.inputs import build_enc_dec_input
from vllm.logger import init_logger
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.renderers.inputs import (
    DecoderDictPrompt,
    DecoderOnlyDictPrompt,
    EncoderDecoderDictPrompt,
    EncoderDictPrompt,
    SingletonDictPrompt,
)
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt, parse_enc_dec_prompt
from vllm.tokenizers import TokenizerLike

from .engine import (
    DecoderEngineInput,
    DecoderOnlyEngineInput,
    EmbedsInput,
    EncoderDecoderInput,
    EncoderInput,
    EngineInput,
    SingletonInput,
    TokensInput,
    tokens_input,
)
from .llm import (
    EmbedsPrompt,
    PromptType,
    TextPrompt,
    TokensPrompt,
)

logger = init_logger(__name__)


class InputPreprocessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
    ) -> None:
        super().__init__()

        self.model_config = vllm_config.model_config
        self.renderer = renderer or renderer_from_config(vllm_config)

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(
            TextPrompt(prompt=prompt),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    def _process_embeds(
        self,
        parsed_content: EmbedsPrompt,
    ) -> EmbedsInput:
        return self.renderer._process_embeds(parsed_content)

    def _truncate_inputs(
        self, inputs: list[int], tokenization_kwargs: dict[str, Any] | None = None
    ) -> list[int]:
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(
            TokensPrompt(prompt_token_ids=inputs),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    def _process_tokens(
        self,
        parsed_content: TokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> TokensInput:
        prompt_token_ids = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs
        )

        inputs = tokens_input(prompt_token_ids)

        if prompt_text := parsed_content.get("prompt"):
            inputs["prompt"] = prompt_text
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_text(
        self,
        parsed_content: TextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> TokensInput:
        prompt_text = parsed_content["prompt"]

        prompt_token_ids = self._tokenize_prompt(
            prompt_text,
            tokenization_kwargs=tokenization_kwargs,
        )
        inputs = tokens_input(prompt_token_ids)

        inputs["prompt"] = prompt_text

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    @overload
    def _prompt_to_llm_inputs(
        self,
        prompt: EncoderDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> EncoderInput: ...

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: DecoderDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> DecoderEngineInput: ...

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: DecoderOnlyDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> DecoderOnlyEngineInput: ...

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> SingletonInput:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt_token_ids" in prompt:
            return self._process_tokens(prompt)  # type: ignore[arg-type]

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
            )

        assert_never(prompt)  # type: ignore[arg-type]

    def _process_encoder_decoder_prompt(
        self,
        prompt: EncoderDecoderDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> EncoderDecoderInput:
        encoder_prompt = prompt["encoder_prompt"]
        decoder_prompt = prompt["decoder_prompt"]

        skip_decoder_start_token = False

        return build_enc_dec_input(
            encoder_input=self._prompt_to_llm_inputs(
                encoder_prompt,
                tokenization_kwargs=tokenization_kwargs,
            ),
            decoder_input=(
                None
                if decoder_prompt is None
                else self._prompt_to_llm_inputs(
                    decoder_prompt,
                    tokenization_kwargs=tokenization_kwargs,
                )
            ),
            decoder_start_token_id=self.renderer.get_dec_start_token_id(),
            skip_decoder_start_token=skip_decoder_start_token,
        )

    def _process_decoder_only_prompt(
        self,
        prompt: DecoderOnlyDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> DecoderOnlyEngineInput:
        return self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
        )

    def preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> EngineInput:
        """Preprocess the input prompt."""
        if self.model_config.is_encoder_decoder:
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder.
            return self._process_encoder_decoder_prompt(
                parse_enc_dec_prompt(prompt),
                tokenization_kwargs,
            )

        return self._process_decoder_only_prompt(
            parse_dec_only_prompt(prompt),
            tokenization_kwargs=tokenization_kwargs,
        )
