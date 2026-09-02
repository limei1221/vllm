# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, overload

from typing_extensions import TypeVar

from vllm.inputs import (
    EmbedsInput,
    EmbedsPrompt,
    EncoderDecoderInput,
    EngineInput,
    SingletonInput,
    TextPrompt,
    TokensInput,
    TokensPrompt,
    build_enc_dec_input,
    embeds_input,
    tokens_input,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.async_utils import make_async
from vllm.utils.torch_utils import set_default_torch_num_threads

from .embed_utils import safe_load_prompt_embeds
from .inputs import (
    DictPrompt,
    EncoderDecoderDictPrompt,
    EncoderDecoderTokPrompt,
    SingletonDictPrompt,
    SingletonTokPrompt,
    TokPrompt,
)
from .inputs.preprocess import extract_target_prompt
from .params import ChatParams, TokenizeParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )

logger = init_logger(__name__)


_T = TypeVar("_T", bound=TokenizerLike, default=TokenizerLike)


class BaseRenderer(ABC, Generic[_T]):
    def __init__(self, config: "VllmConfig", tokenizer: _T | None) -> None:
        super().__init__()

        self.config = config
        self.model_config = config.model_config
        self.api_process_rank = config.parallel_config._api_process_rank

        self.tokenizer = tokenizer

        # Thread pool executor for blocking tokenizer operations.  The
        # processor receives a deep-copied tokenizer (see #36557)
        # so it is safe to run tokenization and MM preprocessing concurrently.
        pool_workers = config.model_config.renderer_num_workers
        self._executor = ThreadPoolExecutor(max_workers=pool_workers)

        # Separate single-worker executor so tokenization never queues behind
        # MM preprocessing; must stay single-worker per #38418 (P0/P1 order).
        self._mm_executor: Executor = ThreadPoolExecutor(max_workers=1)

        # Offload tokenization to the thread pool. The sync
        # ``_tokenize_prompt`` already encapsulates the unified ``__call__``
        # path and char-offset extraction, so the async variant is just it
        # offloaded (mirrors the async processing path below).
        self._tokenize_prompt_async = make_async(
            self._tokenize_prompt, executor=self._executor
        )
        self._async_tokenizer_decode = make_async(self._decode, executor=self._executor)

        self.mm_processor = None
        self._readonly_mm_processor = None
        self._clear_mm_cache_async = make_async(
            self.clear_mm_cache, executor=self._mm_executor
        )
        self._safe_load_prompt_embeds_async = make_async(
            safe_load_prompt_embeds, executor=self._executor
        )

    def get_tokenizer(self) -> _T:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    def _decode(self, *args, **kwargs):
        return self.get_tokenizer().decode(*args, **kwargs)

    @property
    def mm_processor_cache(self) -> None:
        """No multi-modal processor cache exists in this build."""
        return None

    def clear_mm_cache(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.clear_cache()

    def warmup(self, chat_params: ChatParams) -> None:
        """
        Warm up this renderer to avoid first-request latency.

        For chat requests:
        - Jinja2 template compilation
        """
        from vllm.entrypoints.chat_utils import ChatTemplateResolutionError

        # prevent MM processor hangs
        with set_default_torch_num_threads(1):
            try:
                logger.debug("Warming up chat template processing...")
                start_time = time.perf_counter()

                self.render_chat([[{"role": "user", "content": "warmup"}]], chat_params)

                elapsed = time.perf_counter() - start_time
                logger.debug("Chat template warmup completed in %.3fs", elapsed)
            except ChatTemplateResolutionError:
                logger.debug("This model does not support chat template.")
            except Exception:
                logger.warning("Chat template warmup failed", exc_info=True)

    async def clear_mm_cache_async(self) -> None:
        """Serialize clear_mm_cache through the executor to avoid races
        with concurrent process_inputs on the mm_processor_cache."""
        await self._clear_mm_cache_async()

    def shutdown(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.close()

        if executor := getattr(self, "_executor", None):
            executor.shutdown(wait=False)

        if (
            mm_executor := getattr(self, "_mm_executor", None)
        ) is not None and mm_executor is not executor:
            mm_executor.shutdown(wait=False)

    def get_bos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for BOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.bos_token_id

    def get_eos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for EOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.eos_token_id

    def get_dec_start_token_id(self) -> int:
        """
        Obtain the decoder start token id employed by an encoder/decoder model,
        raising an error if it is not available.
        """
        dec_start_token_id = getattr(
            self.model_config.hf_config, "decoder_start_token_id", None
        )

        if dec_start_token_id is None:
            logger.warning_once(
                "Falling back on <BOS> for decoder start token id "
                "because decoder start token id is not available."
            )
            dec_start_token_id = self.get_bos_token_id()

        if dec_start_token_id is None:
            raise RuntimeError("Cannot find decoder start token id or <BOS>")

        return dec_start_token_id

    @cached_property
    def default_cmpl_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor
        if mm_processor is not None:
            return mm_processor.info.default_tok_params

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    @cached_property
    def default_chat_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor
        if mm_processor is not None:
            return mm_processor.info.default_tok_params

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=False,
        )

    # Step 1: Convert raw inputs to prompts
    def render_prompt(
        self,
        prompt: DictPrompt | bytes,
    ) -> DictPrompt:
        if isinstance(prompt, bytes):
            embeds = safe_load_prompt_embeds(self.model_config, prompt)
            prompt = EmbedsPrompt(prompt_embeds=embeds)

        return prompt

    def render_prompts(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return [self.render_prompt(prompt) for prompt in prompts]

    async def _render_prompt_async(
        self,
        prompt: DictPrompt | bytes,
    ) -> DictPrompt:
        if isinstance(prompt, bytes):
            embeds = await self._safe_load_prompt_embeds_async(
                self.model_config, prompt
            )
            return EmbedsPrompt(prompt_embeds=embeds)

        return prompt

    async def render_prompts_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
    ) -> list[DictPrompt]:
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return await asyncio.gather(
            *(self._render_prompt_async(prompt) for prompt in prompts)
        )

    @abstractmethod
    def render_messages(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], DictPrompt]:
        raise NotImplementedError

    async def render_messages_async(
        self,
        messages: list["ChatCompletionMessageParam"],
        params: ChatParams,
    ) -> tuple[list["ConversationMessage"], DictPrompt]:
        return self.render_messages(messages, params)

    # Step 2: Tokenize prompts if necessary
    def _can_produce_offsets(self) -> bool:
        """Whether this renderer's tokenizer can emit char-level offsets.

        Defaults to False; only renderers backed by an HF fast tokenizer
        (see ``HfRenderer``) can produce ``offset_mapping``.
        """
        return False

    def _wants_offsets(
        self,
        prompt: "TextPrompt",
        params: "TokenizeParams",
    ) -> bool:
        return (
            params.return_token_offsets
            and self._can_produce_offsets()
            and not prompt.get("multi_modal_data")
            and not prompt.get("multi_modal_uuids")
        )

    @staticmethod
    def _build_tokens_prompt(
        token_ids: Sequence[int],
        prompt: "TextPrompt",
        *,
        offset_mapping: Sequence[tuple[int, int]] | None = None,
    ) -> "TokensPrompt":
        """Build a TokensPrompt from already-extracted token ids.

        ``offset_mapping`` is the per-token ``(start, end)`` sequence from
        a BatchEncoding; pass it only when offsets were requested, and it
        is attached as ``prompt_token_offsets``.
        """
        if offset_mapping is not None:
            return TokensPrompt(
                prompt_token_ids=list(token_ids),
                prompt_token_offsets=[(int(s), int(e)) for s, e in offset_mapping],
                **prompt,
            )
        return TokensPrompt(prompt_token_ids=list(token_ids), **prompt)

    def _tokenize_prompt(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        want_offsets = self._wants_offsets(prompt, params)
        kwargs = params.get_encode_kwargs()
        if want_offsets:
            kwargs = {**kwargs, "return_offsets_mapping": True}
        encoding = tokenizer(prompt["prompt"], **kwargs)
        return self._build_tokens_prompt(
            encoding["input_ids"],
            prompt,
            offset_mapping=encoding["offset_mapping"] if want_offsets else None,
        )

    def _detokenize_prompt(self, prompt: TokensPrompt) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        prompt["prompt"] = tokenizer.decode(prompt["prompt_token_ids"])

        return prompt

    async def _detokenize_prompt_async(self, prompt: TokensPrompt) -> TokensPrompt:
        prompt["prompt"] = await self._async_tokenizer_decode(
            prompt["prompt_token_ids"]
        )

        return prompt

    @overload
    def _tokenize_singleton_prompt(
        self,
        prompt: TextPrompt | TokensPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    @overload
    def _tokenize_singleton_prompt(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    def _tokenize_singleton_prompt(
        self,
        prompt: SingletonDictPrompt,
        params: TokenizeParams,
    ) -> SingletonTokPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            if not isinstance(prompt.get("prompt"), str):
                raise TypeError(
                    "Expected prompt['prompt'] to be a string before tokenization; "
                    "use 'prompt_token_ids' for token ID inputs"
                )
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = self._tokenize_prompt(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = self._detokenize_prompt(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    @overload
    async def _tokenize_singleton_prompt_async(
        self,
        prompt: TextPrompt | TokensPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    @overload
    async def _tokenize_singleton_prompt_async(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    async def _tokenize_singleton_prompt_async(
        self,
        prompt: SingletonDictPrompt,
        params: TokenizeParams,
    ) -> SingletonTokPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            if not isinstance(prompt.get("prompt"), str):
                raise TypeError(
                    "Expected prompt['prompt'] to be a string before tokenization; "
                    "use 'prompt_token_ids' for token ID inputs"
                )
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = await self._tokenize_prompt_async(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = await self._detokenize_prompt_async(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    def _tokenize_enc_dec_prompt(
        self,
        prompt: EncoderDecoderDictPrompt,
        params: TokenizeParams,
    ) -> EncoderDecoderTokPrompt:
        enc_prompt, dec_prompt = (
            self._tokenize_singleton_prompt(prompt["encoder_prompt"], params),
            (
                None
                if prompt["decoder_prompt"] is None
                else self._tokenize_singleton_prompt(prompt["decoder_prompt"], params)
            ),
        )

        return EncoderDecoderTokPrompt(
            encoder_prompt=enc_prompt,
            decoder_prompt=dec_prompt,
        )

    async def _tokenize_enc_dec_prompt_async(
        self,
        prompt: EncoderDecoderDictPrompt,
        params: TokenizeParams,
    ) -> EncoderDecoderTokPrompt:
        enc_prompt, dec_prompt = await asyncio.gather(
            self._tokenize_singleton_prompt_async(prompt["encoder_prompt"], params),
            (
                asyncio.sleep(0)
                if prompt["decoder_prompt"] is None
                else self._tokenize_singleton_prompt_async(
                    prompt["decoder_prompt"], params
                )
            ),
        )

        return EncoderDecoderTokPrompt(
            encoder_prompt=enc_prompt,
            decoder_prompt=dec_prompt,
        )

    def tokenize_prompt(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            return self._tokenize_enc_dec_prompt(prompt, params)  # type: ignore[arg-type]

        return self._tokenize_singleton_prompt(prompt, params)

    def tokenize_prompts(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return [self.tokenize_prompt(prompt, params) for prompt in prompts]

    async def tokenize_prompt_async(
        self,
        prompt: DictPrompt,
        params: TokenizeParams,
    ) -> TokPrompt:
        if "encoder_prompt" in prompt:
            return await self._tokenize_enc_dec_prompt_async(prompt, params)  # type: ignore[arg-type]

        return await self._tokenize_singleton_prompt_async(prompt, params)

    async def tokenize_prompts_async(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )

    # Step 3: Add extra keys to the prompts
    def _apply_prompt_extras(
        self,
        prompts: Sequence[TokPrompt],
        prompt_extras: dict[str, Any] | None,
    ):
        if not prompt_extras:
            return

        for prompt in prompts:
            target_prompt = extract_target_prompt(self.model_config, prompt)
            target_prompt.update(prompt_extras)  # type: ignore[arg-type]

    # Step 4: Convert to engine inputs
    # TODO: Remove str and tokenization_kwargs after deprecating InputPreprocessor
    def _process_tokens(
        self,
        prompt: TokensPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> TokensInput:
        """Process token inputs, with preprocessing offloaded
        to the shared thread pool in the async variant.
        """
        prompt_token_ids = prompt["prompt_token_ids"]

        engine_input = tokens_input(prompt_token_ids)

        if prompt_text := prompt.get("prompt"):
            engine_input["prompt"] = prompt_text
        if cache_salt := prompt.get("cache_salt"):
            engine_input["cache_salt"] = cache_salt
        # Narrow the union — `prompt_token_offsets` is only on TokensInput.
        if engine_input["type"] == "token" and (
            (offsets := prompt.get("prompt_token_offsets")) is not None
        ):
            engine_input["prompt_token_offsets"] = offsets

        return engine_input

    def _process_embeds(self, prompt: EmbedsPrompt) -> EmbedsInput:
        if not self.model_config.enable_prompt_embeds:
            raise ValueError(
                "You must set `--enable-prompt-embeds` to input `prompt_embeds`."
            )

        prompt_embeds = prompt["prompt_embeds"]

        # prompt_embeds must be (seq_len, hidden_size), but if the user
        # passes in a batch of size 1, i.e. (1, seq_len, hidden_size),
        # we can unambiguously process the intent by squeezing the batch
        # dimension.
        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.squeeze(dim=0)

        if prompt_embeds.ndim != 2:
            raise ValueError("prompt_embeds must be of shape (seq_len, hidden_size).")

        # Tensors must be on CPU for serialization between processes
        # in the MsgpackEncoder. Casting to CPU here ensures that there is no
        # hidden device transfer in the critical path of generation.
        prompt_embeds = prompt_embeds.cpu()

        return embeds_input(
            prompt_embeds=prompt_embeds,
            cache_salt=prompt.get("cache_salt"),
            prompt_token_ids=prompt.get("prompt_token_ids"),
            is_token_ids=prompt.get("prompt_is_token_ids"),
        )

    async def _process_tokens_async(
        self,
        prompt: TokensPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> TokensInput:
        prompt_token_ids = prompt["prompt_token_ids"]

        engine_input = tokens_input(prompt_token_ids)

        if prompt_text := prompt.get("prompt"):
            engine_input["prompt"] = prompt_text
        if cache_salt := prompt.get("cache_salt"):
            engine_input["cache_salt"] = cache_salt
        # Narrow the union — `prompt_token_offsets` is only on TokensInput.
        if engine_input["type"] == "token" and (
            (offsets := prompt.get("prompt_token_offsets")) is not None
        ):
            engine_input["prompt_token_offsets"] = offsets

        return engine_input

    def _process_singleton(
        self,
        prompt: SingletonTokPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> SingletonInput:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        return self._process_tokens(prompt, skip_mm_cache=skip_mm_cache)  # type: ignore[arg-type]

    async def _process_singleton_async(
        self,
        prompt: SingletonTokPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> SingletonInput:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        return await self._process_tokens_async(prompt, skip_mm_cache=skip_mm_cache)  # type: ignore[arg-type]

    def _process_enc_dec(
        self,
        prompt: EncoderDecoderTokPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> EncoderDecoderInput:
        enc_prompt = prompt["encoder_prompt"]
        dec_prompt = prompt["decoder_prompt"]

        skip_decoder_start_token = False

        return build_enc_dec_input(
            encoder_input=self._process_singleton(
                enc_prompt, skip_mm_cache=skip_mm_cache
            ),
            decoder_input=(
                None
                if dec_prompt is None
                else self._process_singleton(dec_prompt, skip_mm_cache=skip_mm_cache)
            ),
            decoder_start_token_id=self.get_dec_start_token_id(),
            skip_decoder_start_token=skip_decoder_start_token,
        )

    async def _process_enc_dec_async(
        self,
        prompt: EncoderDecoderTokPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> EncoderDecoderInput:
        enc_prompt = prompt["encoder_prompt"]
        dec_prompt = prompt["decoder_prompt"]

        encoder_input, decoder_input = await asyncio.gather(
            self._process_singleton_async(enc_prompt, skip_mm_cache=skip_mm_cache),
            (
                asyncio.sleep(0)
                if dec_prompt is None
                else self._process_singleton_async(
                    dec_prompt, skip_mm_cache=skip_mm_cache
                )
            ),
        )

        return build_enc_dec_input(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            decoder_start_token_id=self.get_dec_start_token_id(),
        )

    def process_for_engine(
        self,
        prompt: TokPrompt,
        arrival_time: float,
        *,
        skip_mm_cache: bool = False,
    ) -> EngineInput:
        engine_input: EngineInput
        if "encoder_prompt" in prompt:
            engine_input = self._process_enc_dec(prompt, skip_mm_cache=skip_mm_cache)  # type: ignore[arg-type]
        else:
            engine_input = self._process_singleton(prompt, skip_mm_cache=skip_mm_cache)

        engine_input["arrival_time"] = arrival_time

        return engine_input

    async def process_for_engine_async(
        self,
        prompt: TokPrompt,
        arrival_time: float,
        *,
        skip_mm_cache: bool = False,
    ) -> EngineInput:
        engine_input: EngineInput
        if "encoder_prompt" in prompt:
            engine_input = await self._process_enc_dec_async(
                prompt,  # type: ignore[arg-type]
                skip_mm_cache=skip_mm_cache,
            )
        else:
            engine_input = await self._process_singleton_async(
                prompt, skip_mm_cache=skip_mm_cache
            )

        engine_input["arrival_time"] = arrival_time

        return engine_input

    # Top-level methods
    def render_cmpl(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = self.render_prompts(prompts)
        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return [
            self.process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
            for prompt in tok_prompts
        ]

    async def render_cmpl_async(
        self,
        prompts: Sequence[DictPrompt | bytes],
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = await self.render_prompts_async(prompts)
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return await asyncio.gather(
            *(
                self.process_for_engine_async(
                    p, arrival_time, skip_mm_cache=skip_mm_cache
                )
                for p in tok_prompts
            )
        )

    def render_chat(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in rendered:
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = [
            self.process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
            for prompt in tok_prompts
        ]

        return out_conversations, eng_prompts

    async def render_chat_async(
        self,
        conversations: Sequence[list["ChatCompletionMessageParam"]],
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages_async(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list[list["ConversationMessage"]]()
        dict_prompts = list[DictPrompt]()
        for conv, prompt in await asyncio.gather(*rendered):
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = await asyncio.gather(
            *(
                self.process_for_engine_async(
                    p, arrival_time, skip_mm_cache=skip_mm_cache
                )
                for p in tok_prompts
            )
        )

        return out_conversations, eng_prompts
