# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import types
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.responses import ResponseInputImageParam
from openai_harmony import Message as OpenAIHarmonyMessage
from PIL import Image
from pydantic import BaseModel, ConfigDict, TypeAdapter

# pydantic needs the TypedDict from typing_extensions
from typing_extensions import Required, TypedDict

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import torch
    import transformers
else:
    transformers = LazyLoader("transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)


class ChatTemplateResolutionError(ValueError):
    """Raised when chat template resolution fails.

    This is a subclass of ValueError for backward compatibility with
    existing exception handlers.
    """


MODALITY_PLACEHOLDERS_MAP = {
    "image": "<##IMAGE##>",
    "audio": "<##AUDIO##>",
    "video": "<##VIDEO##>",
    "prompt_embeds": "<##PROMPT_EMBEDS##>",
}


PROMPT_EMBEDS_PLACEHOLDER_TOKEN: Final[str] = "<prompt_embeds>"
"""The special token used as a placeholder for each embedding
position during chat template rendering.

Registered as an additional special token when `--enable-prompt-embeds` is set.
See `_ensure_prompt_embeds_placeholder_token` in `vllm/renderers/hf.py`.
"""


_REQUIRE_MM_PROCESSOR_ERROR: Final[str] = (
    "Resolving modality {modality!r} requires a multimodal processor "
    "but none is available."
)

_ENABLE_PROMPT_EMBEDS_ERROR: Final[str] = (
    "You must set `--enable-prompt-embeds` to input `prompt_embeds`"
)

_PROMPT_EMBEDS_MISSING_DATA_ERROR: Final[str] = (
    "prompt_embeds content part requires a non-empty `data` field "
    "with base64-encoded tensor bytes."
)

_RESERVED_PLACEHOLDER_IN_TEXT_ERROR: Final[str] = (
    "Text content may not contain the reserved placeholder {token!r}. "
    "This placeholder is used internally to mark `prompt_embeds` splice "
    "positions in the tokenized prompt."
)


class AudioURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the audio or a data URL with base64 encoded audio data.
    """


class ChatCompletionContentPartAudioParam(TypedDict, total=False):
    audio_url: Required[AudioURL]

    type: Required[Literal["audio_url"]]
    """The type of the content part."""


class ChatCompletionContentPartImageEmbedsParam(TypedDict, total=False):
    image_embeds: str | dict[str, str] | None
    """
    The image embeddings. It can be either:
    - A single base64 string.
    - A dictionary where each value is a base64 string.
    """
    type: Required[Literal["image_embeds"]]
    """The type of the content part."""
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class ChatCompletionContentPartAudioEmbedsParam(TypedDict, total=False):
    audio_embeds: str | dict[str, str] | None
    """
    The audio embeddings. It can be either:
    - A single base64 string representing a serialized torch tensor.
    - A dictionary where each value is a base64 string.
    """
    type: Required[Literal["audio_embeds"]]
    """The type of the content part."""
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class ChatCompletionContentPartPromptEmbedsParam(TypedDict, total=False):
    data: Required[str]
    """
    Base64-encoded bytes of a serialized `torch.Tensor` of shape
    `(num_tokens, hidden_size)`. The tensor's `dtype` and `hidden_size` must
    match the model's input embedding layer.
    """
    type: Required[Literal["prompt_embeds"]]
    """The type of the content part."""


class VideoURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the video or a data URL with base64 encoded video data.
    """


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    video_url: Required[VideoURL]

    type: Required[Literal["video_url"]]
    """The type of the content part."""


class PILImage(BaseModel):
    """
    A PIL.Image.Image object.
    """

    image_pil: Image.Image
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CustomChatCompletionContentPILImageParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a PIL image.

    Example:
    {
        "image_pil": ImageAsset('cherry_blossom').pil_image
    }
    """

    image_pil: PILImage | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class CustomChatCompletionContentSimpleImageParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain image_url.
    This is supported by OpenAI API, although it is not documented.

    Example:
    {
        "image_url": "https://example.com/image.jpg"
    }
    """

    image_url: str | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class CustomChatCompletionContentSimpleAudioParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "audio_url": "https://example.com/audio.mp3"
    }
    """

    audio_url: str | None


class CustomChatCompletionContentSimpleVideoParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "video_url": "https://example.com/video.mp4"
    }
    """

    video_url: str | None
    uuid: str | None
    """
    User-provided UUID of a media. User must guarantee that it is properly
    generated and unique for different medias.
    """


class CustomThinkCompletionContentParam(TypedDict, total=False):
    """A Think Completion Content Param that accepts a plain text and a boolean.

    Example:
    {
        "thinking": "I am thinking about the answer",
        "closed": True,
        "type": "thinking"
    }
    """

    thinking: Required[str]
    """The thinking content."""

    closed: bool
    """Whether the thinking is closed."""

    type: Required[Literal["thinking"]]
    """The thinking type."""


class CustomChatCompletionContentToolReferenceParam(TypedDict, total=False):
    """A tool reference content param that only accepts a plain tool name.

    Example:
    {
        "name": "get_weather",
        "type": "tool_reference"
    }
    """

    name: str
    """The name of the tool being referenced."""

    type: Literal["tool_reference"]
    """The content type."""


ChatCompletionContentPartParam: TypeAlias = (
    OpenAIChatCompletionContentPartParam
    | ChatCompletionContentPartAudioParam
    | ChatCompletionContentPartInputAudioParam
    | ChatCompletionContentPartVideoParam
    | ChatCompletionContentPartRefusalParam
    | CustomChatCompletionContentPILImageParam
    | CustomChatCompletionContentSimpleImageParam
    | ChatCompletionContentPartImageEmbedsParam
    | ChatCompletionContentPartAudioEmbedsParam
    | ChatCompletionContentPartPromptEmbedsParam
    | CustomChatCompletionContentSimpleAudioParam
    | CustomChatCompletionContentSimpleVideoParam
    | CustomChatCompletionContentToolReferenceParam
    | str
    | CustomThinkCompletionContentParam
)


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""

    role: Required[str]
    """The role of the message's author."""

    content: str | list[ChatCompletionContentPartParam]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """

    tool_call_id: str | None
    """Tool call that this message is responding to."""

    tool_calls: list[ChatCompletionMessageToolCallParam] | None
    """The tool calls generated by the model, such as function calls."""

    reasoning: str | None
    """The reasoning content for interleaved thinking."""

    tools: list[ChatCompletionFunctionToolParam] | None
    """The tools for developer role."""

    task: str | None
    """Model-specific task marker. Currently passed through for DeepSeek V4."""


ChatCompletionMessageParam: TypeAlias = (
    OpenAIChatCompletionMessageParam
    | CustomChatCompletionMessageParam
    | OpenAIHarmonyMessage
)


# TODO: Make fields ReadOnly once mypy supports it
class ConversationMessage(TypedDict, total=False):
    role: Required[str]
    """The role of the message's author."""

    content: str | None | list[dict[str, str]]
    """The contents of the message"""

    tool_call_id: str | None
    """Tool call that this message is responding to."""

    name: str | None
    """The name of the function to call"""

    tool_calls: list[ChatCompletionMessageToolCallParam] | None
    """The tool calls generated by the model, such as function calls."""

    reasoning: str | None
    """The reasoning content for interleaved thinking."""

    reasoning_content: str | None
    """Deprecated: The reasoning content for interleaved thinking."""

    tools: list[ChatCompletionFunctionToolParam] | None
    """The tools for developer role."""

    task: str | None
    """Model-specific task marker. Currently passed through for DeepSeek V4."""


# Passed in by user
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]

# After resolving "auto"
ChatTemplateContentFormat = Literal["string", "openai"]


ModalityStr = Literal[
    "image",
    "audio",
    "video",
    "image_embeds",
    "audio_embeds",
    "vision_chunk",
    "prompt_embeds",
]
_T = TypeVar("_T")
_AsyncMultiModalItem: TypeAlias = Callable[[], Awaitable[tuple[object, str | None]]]


# Backward compatibility for single item input
@dataclass
class ChatTemplateConfig:
    chat_template: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    trust_request_chat_template: bool = False


def validate_chat_template(chat_template: Path | str | None):
    """Raises if the provided chat template appears invalid."""
    if chat_template is None:
        return

    elif isinstance(chat_template, Path) and not chat_template.exists():
        raise FileNotFoundError("the supplied chat template path doesn't exist")

    elif isinstance(chat_template, str):
        JINJA_CHARS = "{}\n"
        if (
            not any(c in chat_template for c in JINJA_CHARS)
            and not Path(chat_template).exists()
        ):
            # Try to find the template in the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            if not builtin_template_path.exists():
                raise ValueError(
                    f"The supplied chat template string ({chat_template}) "
                    f"appears path-like, but doesn't exist! "
                    f"Tried: {chat_template} and {builtin_template_path}"
                )

    else:
        raise TypeError(f"{type(chat_template)} is not a valid chat template type")


def _load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    if chat_template is None:
        return None

    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError(
                "chat_template is expected to be read directly from its value"
            )

        return chat_template

    try:
        with open(chat_template) as f:
            return f.read()
    except OSError as e:
        if isinstance(chat_template, Path):
            raise

        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            # Try to load from the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            try:
                with open(builtin_template_path) as f:
                    return f.read()
            except OSError:
                msg = (
                    f"The supplied chat template ({chat_template}) "
                    f"looks like a file path, but it failed to be opened. "
                    f"Tried: {chat_template} and {builtin_template_path}. "
                    f"Reason: {e}"
                )
                raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        return _load_chat_template(chat_template, is_literal=True)


_cached_load_chat_template = lru_cache(_load_chat_template)


def load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    return _cached_load_chat_template(chat_template, is_literal=is_literal)


def _get_interleaved_text_prompt(
    placeholder_storage: dict[str, list], texts: list[str]
) -> str:
    for idx, elem in enumerate(texts):
        if elem in placeholder_storage:
            texts[idx] = placeholder_storage[elem].pop(0)

    return "\n".join(texts)


# TODO: Let user specify how to insert multimodal tokens into prompt
# (similar to chat template)
def _get_full_multimodal_text_prompt(
    placeholder_storage: dict[str, list],
    texts: list[str],
    interleave_strings: bool,
    multimodal_content_part_separator: str = "\n",
) -> str:
    """Combine multimodal prompts for a multimodal language model."""

    # flatten storage to make it looks like
    # {
    #   "<|image|>": 2,
    #   "<|audio|>": 1
    # }
    placeholder_counts = Counter(
        [v for elem in placeholder_storage.values() for v in elem]
    )

    if interleave_strings:
        text_prompt = _get_interleaved_text_prompt(placeholder_storage, texts)
    else:
        text_prompt = "\n".join(texts)

    # Pass interleaved text further in case the user used image placeholders
    # himself, but forgot to disable the 'interleave_strings' flag

    # Look through the text prompt to check for missing placeholders
    missing_placeholders: list[str] = []
    for placeholder in placeholder_counts:
        # For any existing placeholder in the text prompt, we leave it as is
        placeholder_counts[placeholder] -= text_prompt.count(placeholder)

        if placeholder_counts[placeholder] < 0:
            logger.error(
                "Placeholder count is negative! "
                "Ensure that the 'interleave_strings' flag is disabled "
                "(current value: %s) "
                "when manually placing image placeholders.",
                interleave_strings,
            )
            logger.debug("Input prompt: %s", text_prompt)
            raise VLLMValidationError(
                f"Found more '{placeholder}' placeholders in input prompt than "
                "actual multimodal data items."
            )

        missing_placeholders.extend([placeholder] * placeholder_counts[placeholder])

    # NOTE: Default behaviour: we always add missing placeholders
    # at the front of the prompt, if interleave_strings=False
    if text_prompt:
        return multimodal_content_part_separator.join(
            missing_placeholders + [text_prompt]
        )
    else:
        return multimodal_content_part_separator.join(missing_placeholders)


# No need to validate using Pydantic again
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageEmbedsParser = partial(cast, ChatCompletionContentPartImageEmbedsParam)
_AudioEmbedsParser = partial(cast, ChatCompletionContentPartAudioEmbedsParam)
_PromptEmbedsParser = partial(cast, ChatCompletionContentPartPromptEmbedsParam)
_InputAudioParser = partial(cast, ChatCompletionContentPartInputAudioParam)
_RefusalParser = partial(cast, ChatCompletionContentPartRefusalParam)
_PILImageParser = partial(cast, CustomChatCompletionContentPILImageParam)
_ThinkParser = partial(cast, CustomThinkCompletionContentParam)
# Need to validate url objects
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam).validate_python
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam).validate_python
_VideoParser = TypeAdapter(ChatCompletionContentPartVideoParam).validate_python

_ResponsesInputImageParser = TypeAdapter(ResponseInputImageParam).validate_python
_ContentPart: TypeAlias = str | dict[str, str] | InputAudio | PILImage

# Define a mapping from part types to their corresponding parsing functions.
MM_PARSER_MAP: dict[
    str,
    Callable[[ChatCompletionContentPartParam], _ContentPart],
] = {
    "text": lambda part: _TextParser(part).get("text", None),
    "thinking": lambda part: _ThinkParser(part).get("thinking", None),
    "input_text": lambda part: _TextParser(part).get("text", None),
    "output_text": lambda part: _TextParser(part).get("text", None),
    "input_image": lambda part: _ResponsesInputImageParser(part).get("image_url", None),
    "image_url": lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
    "image_embeds": lambda part: _ImageEmbedsParser(part).get("image_embeds", None),
    "audio_embeds": lambda part: _AudioEmbedsParser(part).get("audio_embeds", None),
    "prompt_embeds": lambda part: _PromptEmbedsParser(part).get("data", None),
    "image_pil": lambda part: _PILImageParser(part).get("image_pil", None),
    "audio_url": lambda part: _AudioParser(part).get("audio_url", {}).get("url", None),
    "input_audio": lambda part: _InputAudioParser(part).get("input_audio", None),
    "refusal": lambda part: _RefusalParser(part).get("refusal", None),
    "video_url": lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
    "tool_reference": lambda part: cast(
        CustomChatCompletionContentToolReferenceParam, part
    ).get("name", None),
}


def _collect_known_content_part_fields() -> frozenset[str]:
    fields: set[str] = set()
    stack: list[Any] = [ChatCompletionContentPartParam]
    while stack:
        node = stack.pop()
        if get_origin(node) in (Union, types.UnionType):
            stack.extend(get_args(node))
        elif hasattr(node, "__required_keys__"):
            fields |= node.__required_keys__ | node.__optional_keys__
    return frozenset(fields)


_KNOWN_CONTENT_PART_FIELDS = _collect_known_content_part_fields()


def _collect_extra_fields(part: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in part.items() if k not in _KNOWN_CONTENT_PART_FIELDS}


def _parse_chat_message_content_mm_part(
    part: ChatCompletionContentPartParam,
) -> tuple[str, _ContentPart]:
    """
    Parses a given multi-modal content part based on its type.

    Args:
        part: A dict containing the content part, with a potential 'type' field.

    Returns:
        A tuple (part_type, content) where:
        - part_type: Type of the part (e.g., 'text', 'image_url').
        - content: Parsed content (e.g., text, image URL).

    Raises:
        ValueError: If the 'type' field is missing and no direct URL is found.
    """
    assert isinstance(
        part, dict
    )  # This is needed to avoid mypy errors: part.get() from str
    part_type = part.get("type", None)
    uuid = part.get("uuid", None)

    if isinstance(part_type, str) and part_type in MM_PARSER_MAP and uuid is None:  # noqa: E501
        content = MM_PARSER_MAP[part_type](part)

        # Special case for 'image_url.detail'
        # We only support 'auto', which is the default
        if part_type == "image_url" and part.get("detail", "auto") != "auto":
            logger.warning(
                "'image_url.detail' is currently not supported and will be ignored."
            )

        return part_type, content

    # Handle missing 'type' but provided direct URL fields.
    # 'type' is required field by pydantic
    if part_type is None or uuid is not None:
        if "image_url" in part:
            image_params = cast(CustomChatCompletionContentSimpleImageParam, part)
            image_url = image_params.get("image_url", None)
            if isinstance(image_url, dict):
                # Can potentially happen if user provides a uuid
                # with url as a dict of {"url": url}
                image_url = image_url.get("url", None)
            return "image_url", image_url
        if "image_pil" in part:
            # "image_pil" could be None if UUID is provided.
            image_params = cast(  # type: ignore
                CustomChatCompletionContentPILImageParam, part
            )
            image_pil = image_params.get("image_pil", None)
            return "image_pil", image_pil
        if "image_embeds" in part:
            # "image_embeds" could be None if UUID is provided.
            image_params = cast(  # type: ignore
                ChatCompletionContentPartImageEmbedsParam, part
            )
            image_embeds = image_params.get("image_embeds", None)
            return "image_embeds", image_embeds
        if "audio_embeds" in part:
            # "audio_embeds" could be None if UUID is provided.
            audio_params = cast(  # type: ignore[assignment]
                ChatCompletionContentPartAudioEmbedsParam, part
            )
            audio_embeds = audio_params.get("audio_embeds", None)
            return "audio_embeds", audio_embeds
        if "prompt_embeds" in part:
            prompt_embeds_params = cast(  # type: ignore[assignment]
                ChatCompletionContentPartPromptEmbedsParam, part
            )
            return "prompt_embeds", prompt_embeds_params.get("data", None)
        if "audio_url" in part:
            audio_params = cast(  # type: ignore[assignment]
                CustomChatCompletionContentSimpleAudioParam, part
            )
            audio_url = audio_params.get("audio_url", None)
            if isinstance(audio_url, dict):
                # Can potentially happen if user provides a uuid
                # with url as a dict of {"url": url}
                audio_url = audio_url.get("url", None)
            return "audio_url", audio_url
        if part.get("input_audio") is not None:
            input_audio_params = _InputAudioParser(part).get("input_audio", None)
            return "input_audio", input_audio_params
        if "video_url" in part:
            video_params = cast(CustomChatCompletionContentSimpleVideoParam, part)
            video_url = video_params.get("video_url", None)
            if isinstance(video_url, dict):
                # Can potentially happen if user provides a uuid
                # with url as a dict of {"url": url}
                video_url = video_url.get("url", None)
            return "video_url", video_url
        if "tool_reference" in part:
            tool_reference_params = cast(
                CustomChatCompletionContentToolReferenceParam, part
            )
            tool_reference = tool_reference_params.get("name", None)
            return "tool_reference", tool_reference
        # Raise an error if no 'type' or direct URL is found.
        raise VLLMValidationError(
            "Missing 'type' field in multimodal part.", parameter="type"
        )

    if not isinstance(part_type, str):
        raise VLLMValidationError(
            "Invalid 'type' field in multimodal part.", parameter="type"
        )
    return part_type, "unknown part_type content"


PART_TYPES_TO_SKIP_NONE_CONTENT = (
    "text",
    "refusal",
)


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    model_config: ModelConfig,
    *,
    wrap_dicts: bool,
) -> list[ConversationMessage]:
    content = list[_ContentPart]()

    for part in parts:
        parse_res = _parse_chat_message_content_part(
            part,
            model_config,
            wrap_dicts=wrap_dicts,
        )
        if parse_res:
            content.append(parse_res)

    if wrap_dicts:
        return [ConversationMessage(role=role, content=content)]  # type: ignore
    texts = cast(list[str], content)
    return [ConversationMessage(role=role, content="\n".join(texts))]


def _reject_reserved_placeholder_in_text(text: str, model_config: ModelConfig) -> None:
    """Reject user-supplied text parts that contains the reserved `prompt_embeds`
    placeholder sentinel.

    When the server accepts `prompt_embeds`, the placeholder token is
    registered as a single unsplittable special token on the tokenizer. Any
    user text that happens to contain the literal sequence would tokenize to
    the same ID and be mistaken for a splice point by the renderer, letting a
    caller move or inject splice positions via plain text content.
    """
    if model_config.enable_prompt_embeds and PROMPT_EMBEDS_PLACEHOLDER_TOKEN in text:
        raise VLLMValidationError(
            _RESERVED_PLACEHOLDER_IN_TEXT_ERROR.format(
                token=PROMPT_EMBEDS_PLACEHOLDER_TOKEN
            )
        )


def _parse_chat_message_content_part(
    part: ChatCompletionContentPartParam,
    model_config: ModelConfig,
    *,
    wrap_dicts: bool,
) -> _ContentPart | None:
    """Parse a single content part of a conversation.

    This build serves text only, so media parts are rejected rather than
    routed to a multi-modal parser.
    """
    if isinstance(part, str):  # Handle plain text parts
        _reject_reserved_placeholder_in_text(part, model_config)
        if wrap_dicts:
            return {"type": "text", "text": part}
        return part

    part_type, content = _parse_chat_message_content_mm_part(part)
    if part_type in PART_TYPES_TO_SKIP_NONE_CONTENT and content is None:
        logger.warning(
            "Skipping content part '%s' (type: '%s') with empty content.",
            part,
            part_type,
        )
        return None

    if part_type in ("text", "input_text", "output_text", "refusal", "thinking"):
        str_content = cast(str, content)
        _reject_reserved_placeholder_in_text(str_content, model_config)
        if wrap_dicts:
            result: dict[str, Any] = {"type": "text", "text": str_content}
            result.update(_collect_extra_fields(cast(dict[str, Any], part)))
            return result
        return str_content

    if part_type == "tool_reference":
        # Tool references are passed through for the chat template to expand.
        if wrap_dicts:
            return {"type": "tool_reference", "name": cast(str, content)}
        return cast(str, content)

    raise VLLMValidationError(
        f"Unsupported chat content part type: {part_type!r}. "
        "This build supports text content only.",
        parameter="type",
        value=part_type,
    )


# No need to validate using Pydantic again
_AssistantParser = partial(cast, ChatCompletionAssistantMessageParam)
_ToolParser = partial(cast, ChatCompletionToolMessageParam)


def _parse_chat_message_content(
    message: ChatCompletionMessageParam,
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
) -> list[ConversationMessage]:
    role = message["role"]
    content = message.get("content")
    reasoning = message.get("reasoning")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [ChatCompletionContentPartTextParam(type="text", text=content)]
    result = _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        model_config,
        wrap_dicts=(content_format == "openai"),
    )

    for result_msg in result:
        if role == "assistant":
            parsed_msg = _AssistantParser(message)

            # The 'tool_calls' is not None check ensures compatibility.
            # It's needed only if downstream code doesn't strictly
            # follow the OpenAI spec.
            if "tool_calls" in parsed_msg and parsed_msg["tool_calls"] is not None:
                result_msg["tool_calls"] = list(parsed_msg["tool_calls"])
            # Include reasoning if present for interleaved thinking.
            if reasoning is not None:
                result_msg["reasoning"] = cast(str, reasoning)
                result_msg["reasoning_content"] = cast(
                    str, reasoning
                )  # keep compatibility
        elif role == "tool":
            parsed_msg = _ToolParser(message)
            if "tool_call_id" in parsed_msg:
                result_msg["tool_call_id"] = parsed_msg["tool_call_id"]
            # Normalize tool message content from OpenAI array format to plain
            # string. Clients like Claude Code / Cursor send tool results as
            # [{"type": "text", "text": "..."}], but most chat templates only
            # handle string content for tool messages.
            # However, tool_reference items must be preserved as structured
            # dicts for the chat template to expand them.
            msg_content = result_msg.get("content")
            if isinstance(msg_content, list):
                has_non_text = any(
                    isinstance(item, dict) and item.get("type") != "text"
                    for item in msg_content
                )
                if has_non_text:
                    # Keep structured content (e.g., tool_reference)
                    result_msg["content"] = msg_content
                else:
                    texts = [
                        item.get("text", "")
                        for item in msg_content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    result_msg["content"] = "\n".join(texts) if texts else ""

        if "name" in message and isinstance(message["name"], str):
            result_msg["name"] = message["name"]

        if "task" in message and isinstance(message["task"], str):
            result_msg["task"] = message["task"]

        if role == "developer":
            result_msg["tools"] = message.get("tools", None)
    return result


def _postprocess_messages(messages: list[ConversationMessage]) -> None:
    # per the Transformers docs & maintainers, tool call arguments in
    # assistant-role messages with tool_calls need to be dicts not JSON str -
    # this is how tool-use chat templates will expect them moving forwards
    # so, for messages that have tool_calls, parse the string (which we get
    # from openAI format) to dict
    for message in messages:
        if message["role"] == "assistant" and "tool_calls" in message:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            if len(tool_calls) == 0:
                # Drop empty tool_calls to keep templates on the normal assistant path.
                message.pop("tool_calls", None)
                continue

            for item in tool_calls:
                if not isinstance(item, dict):
                    raise VLLMValidationError(
                        "assistant tool_calls entries must be objects.",
                        parameter="tool_calls",
                    )

                function = item.get("function")
                if item.get("type", "function") != "function" or not isinstance(
                    function, dict
                ):
                    raise VLLMValidationError(
                        "chat completions only support assistant tool_calls "
                        "of type 'function'.",
                        parameter="tool_calls",
                    )

                # if arguments is None or empty string, set to {}
                if content := function.get("arguments"):
                    if not isinstance(content, (dict, list)):
                        parsed = json.loads(content)
                        function["arguments"] = parsed if parsed is not None else {}
                else:
                    function["arguments"] = {}


def parse_chat_messages(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
    media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> tuple[list[ConversationMessage], None, None]:
    """Parse chat messages into a conversation. Text-only in this build."""
    del media_io_kwargs, mm_processor_kwargs
    conversation: list[ConversationMessage] = []

    for msg in messages:
        conversation.extend(
            _parse_chat_message_content(msg, model_config, content_format)
        )

    _postprocess_messages(conversation)

    return conversation, None, None


async def parse_chat_messages_async(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
    media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> tuple[list[ConversationMessage], None, None]:
    """Async variant of :func:`parse_chat_messages`. Text-only in this build."""
    del media_io_kwargs, mm_processor_kwargs
    conversation: list[ConversationMessage] = []

    for msg in messages:
        conversation.extend(
            _parse_chat_message_content(msg, model_config, content_format)
        )

    _postprocess_messages(conversation)

    return conversation, None, None


_KIMI_MODEL_TYPES = ("kimi_k2", "kimi_k25", "kimi_k3")


def get_tool_call_id_type(model_config: ModelConfig) -> str:
    """Return the tool-call ID type for a given model configuration."""
    hf_overrides = getattr(model_config, "hf_overrides", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    model_types = (
        getattr(hf_config, "model_type", None),
        getattr(hf_text_config, "model_type", None),
    )
    if any(model_type in _KIMI_MODEL_TYPES for model_type in model_types) or (
        isinstance(hf_overrides, dict)
        and hf_overrides.get("model_type") in _KIMI_MODEL_TYPES
    ):
        return "kimi_k2"
    return "random"


def make_tool_call_id(id_type: str = "random", func_name=None, idx=None):
    if id_type == "kimi_k2":
        return f"functions.{func_name}:{idx}"
    else:
        # by default return random
        return f"chatcmpl-tool-{random_uuid()}"
