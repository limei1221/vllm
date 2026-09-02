"""Schema and utilities for inputs to the engine client (`LLMEngine`/`AsyncLLM`)."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Literal, TypeAlias

from typing_extensions import NotRequired, TypedDict, assert_never

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    import torch


class _InputOptions(TypedDict):
    """
    Additional options available to all
    [`SingletonInput`][vllm.inputs.engine.SingletonInput] types.
    """

    arrival_time: NotRequired[float]
    """The time when the input was received (before rendering)."""

    cache_salt: NotRequired[str]
    """Optional cache salt to be used for prefix caching."""


class TokensInput(_InputOptions):
    """Represents token-based input to the engine."""

    type: Literal["token"]
    """The type of input."""

    prompt_token_ids: list[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    prompt_token_offsets: NotRequired[list[tuple[int, int]] | None]
    """Char-level (start, end) offsets per token, propagated from the
    renderer's TokensPrompt when offsets were computed."""

    assistant_tokens_mask: NotRequired[list[int] | None]
    """Per-token 0/1 mask marking assistant-generated tokens.
    Populated when ``return_assistant_tokens_mask=True`` is set on the
    render request and the chat template supports ``{% generation %}``."""


def tokens_input(
    prompt_token_ids: list[int],
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> TokensInput:
    """
    Construct [`TokensInput`][vllm.inputs.engine.TokensInput]
    from optional values.
    """
    inputs = TokensInput(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


class EmbedsInput(_InputOptions):
    """Represents embeddings-based input to the engine."""

    type: Literal["embeds"]
    """The type of input."""

    prompt_embeds: "torch.Tensor"
    """The embeddings of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    prompt_token_ids: NotRequired[list[int]]
    """Token IDs of the rendered prompt. Only set for mixed-mode inputs
    (chat completion with `prompt_embeds` content parts). When present,
    `is_token_ids` MUST also be present and have the same length. 
    For pure-embeds inputs this field is absent."""

    is_token_ids: NotRequired[list[bool]]
    """Per-position mask for mixed-mode inputs. `True` means the position
    is a real token ID (use the model's embedding layer); `False` means
    the position uses a pre-computed embedding row from `prompt_embeds`.
    Length MUST equal `len(prompt_token_ids)`.
    For pure-embeds inputs this field is absent."""


def embeds_input(
    prompt_embeds: "torch.Tensor",
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
    prompt_token_ids: list[int] | None = None,
    is_token_ids: list[bool] | None = None,
) -> EmbedsInput:
    """
    Construct [`EmbedsInput`][vllm.inputs.engine.EmbedsInput]
    from optional values.
    """
    inputs = EmbedsInput(type="embeds", prompt_embeds=prompt_embeds)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
    if prompt_token_ids is not None:
        inputs["prompt_token_ids"] = prompt_token_ids
    if is_token_ids is not None:
        inputs["is_token_ids"] = is_token_ids

    return inputs


DecoderOnlyEngineInput: TypeAlias = TokensInput | EmbedsInput
"""
A rendered [`DecoderOnlyPrompt`][vllm.inputs.llm.DecoderOnlyPrompt]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


EncoderInput: TypeAlias = TokensInput
"""
A rendered [`EncoderPrompt`][vllm.inputs.llm.EncoderPrompt]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


DecoderEngineInput: TypeAlias = TokensInput
"""
A rendered [`DecoderPrompt`][vllm.inputs.llm.DecoderPrompt]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


class EncoderDecoderInput(TypedDict):
    """
    A rendered [`EncoderDecoderPrompt`][vllm.inputs.llm.EncoderDecoderPrompt]
    which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
    """

    type: Literal["enc_dec"]

    encoder_prompt: EncoderInput
    """The inputs for the encoder portion."""

    decoder_prompt: DecoderEngineInput
    """The inputs for the decoder portion."""

    arrival_time: NotRequired[float]
    """The time when the input was received (before rendering)."""


SingletonInput: TypeAlias = DecoderOnlyEngineInput
"""
A rendered [`SingletonPrompt`][vllm.inputs.llm.SingletonPrompt]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


EngineInput: TypeAlias = DecoderOnlyEngineInput | EncoderDecoderInput
"""
A rendered [`PromptType`][vllm.inputs.llm.PromptType]
which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
"""


def _validate_enc_input(enc_input: SingletonInput) -> EncoderInput:
    if enc_input["type"] == "embeds":
        raise VLLMValidationError(
            "Embedding inputs are not supported for encoder-decoder models"
        )

    return enc_input  # type: ignore[return-value]


def _validate_dec_input(dec_input: SingletonInput) -> DecoderEngineInput:
    if dec_input["type"] == "embeds":
        raise VLLMValidationError(
            "Embedding inputs are not supported for encoder-decoder models"
        )

    return dec_input


def _prepare_decoder_input_ids_for_generation(
    decoder_input_ids: list[int],
    decoder_start_token_id: int,
) -> list[int]:
    """
    Prepare `decoder_input_ids` for generation with encoder-decoder models,
    according to `GenerationMixin._prepare_decoder_input_ids_for_generation()`.

    Source:
    https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/generation/utils.py
    """
    if len(decoder_input_ids) == 0 or decoder_input_ids[0] != decoder_start_token_id:
        decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

    return decoder_input_ids


def build_enc_dec_input(
    encoder_input: SingletonInput,
    decoder_input: SingletonInput | None,
    decoder_start_token_id: int,
    skip_decoder_start_token: bool = False,
) -> EncoderDecoderInput:
    enc_input = _validate_enc_input(encoder_input)

    if decoder_input is None:
        dec_input: DecoderEngineInput = enc_input
    else:
        dec_input = _validate_dec_input(decoder_input)

    enc_input_new: EncoderInput
    dec_input_new: DecoderEngineInput

    if enc_input["type"] == "token":
        enc_input_new = tokens_input(prompt_token_ids=[])
        dec_input_new = dec_input
    else:
        assert_never(enc_input)

    if not skip_decoder_start_token:
        dec_input_new["prompt_token_ids"] = _prepare_decoder_input_ids_for_generation(
            dec_input_new["prompt_token_ids"],
            decoder_start_token_id,
        )

    if cache_salt := enc_input.get("cache_salt"):
        dec_input_new["cache_salt"] = cache_salt

    return EncoderDecoderInput(
        type="enc_dec",
        encoder_prompt=enc_input_new,
        decoder_prompt=dec_input_new,
    )


def split_enc_dec_input(
    inputs: EngineInput,
) -> tuple[SingletonInput | None, SingletonInput]:
    if inputs["type"] == "enc_dec":
        return inputs["encoder_prompt"], inputs["decoder_prompt"]

    return None, inputs
