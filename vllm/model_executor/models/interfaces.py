# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import (
    AsyncGenerator,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    overload,
    runtime_checkable,
)

import numpy as np
import torch
from torch import Tensor
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from typing_extensions import Self, TypeIs

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import (
        ModelConfig,
    )
    from vllm.inputs import PromptType
    from vllm.model_executor.layers.fused_moe import MoERunner
    from vllm.model_executor.models.utils import WeightsMapper

    _ProcessorFactories: TypeAlias = Any
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


def requires_raw_input_tokens(model: type[object] | object) -> bool:
    return getattr(model, "requires_raw_input_tokens", False)


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
class _MakeEmptyIntermediateTensors(Protocol):
    def __call__(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "IntermediateTensors": ...


@runtime_checkable
class SupportsPP(Protocol):
    """The interface required for all models that support pipeline parallel."""

    supports_pp: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports pipeline parallel.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    make_empty_intermediate_tensors: _MakeEmptyIntermediateTensors
    """Called when PP rank > 0 for profiling purposes."""

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        *,
        intermediate_tensors: "IntermediateTensors | None",
    ) -> "Tensor | IntermediateTensors | tuple[Tensor, list[Tensor]]":
        """
        Accept [`IntermediateTensors`][vllm.sequence.IntermediateTensors] when
        PP rank > 0.

        Return [`IntermediateTensors`][vllm.sequence.IntermediateTensors] only
        for the last PP rank.
        """
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsPPType(Protocol):
    supports_pp: Literal[True]

    make_empty_intermediate_tensors: _MakeEmptyIntermediateTensors

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        *,
        intermediate_tensors: "IntermediateTensors | None",
    ) -> "Tensor | IntermediateTensors | tuple[Tensor, list[Tensor]]": ...


@overload
def supports_pp(model: type[object]) -> TypeIs[type[SupportsPP]]: ...


@overload
def supports_pp(model: object) -> TypeIs[SupportsPP]: ...


def supports_pp(
    model: type[object] | object,
) -> bool | TypeIs[type[SupportsPP]] | TypeIs[SupportsPP]:
    supports_attributes = _supports_pp_attributes(model)
    supports_inspect = _supports_pp_inspect(model)

    if supports_attributes and not supports_inspect:
        logger.warning(
            "The model (%s) sets `supports_pp=True`, but does not accept "
            "`intermediate_tensors` in its `forward` method",
            model,
        )

    if not supports_attributes:
        pp_attrs = ("make_empty_intermediate_tensors",)
        missing_attrs = tuple(attr for attr in pp_attrs if not hasattr(model, attr))

        if getattr(model, "supports_pp", False):
            if missing_attrs:
                logger.warning(
                    "The model (%s) sets `supports_pp=True`, "
                    "but is missing PP-specific attributes: %s",
                    model,
                    missing_attrs,
                )
        else:
            if not missing_attrs:
                logger.warning(
                    "The model (%s) contains all PP-specific attributes, "
                    "but does not set `supports_pp=True`.",
                    model,
                )

    return supports_attributes and supports_inspect


def _supports_pp_attributes(model: type[object] | object) -> bool:
    if isinstance(model, type):
        return SupportsPP in model.__mro__ or isinstance(model, _SupportsPPType)

    return isinstance(model, SupportsPP)


def _supports_pp_inspect(model: type[object] | object) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    return supports_kw(model_forward, "intermediate_tensors")


@runtime_checkable
class HasInnerState(Protocol):
    """The interface required for all models that has inner state."""

    has_inner_state: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has inner state.
        Models that has inner state usually need access to the scheduler_config
        for max_num_seqs, etc. True for e.g. both Mamba and Jamba.
    """


@overload
def has_inner_state(model: object) -> TypeIs[HasInnerState]: ...


@overload
def has_inner_state(model: type[object]) -> TypeIs[type[HasInnerState]]: ...


def has_inner_state(
    model: type[object] | object,
) -> TypeIs[type[HasInnerState]] | TypeIs[HasInnerState]:
    return getattr(model, "has_inner_state", False)


@runtime_checkable
class MixtureOfExperts(Protocol):
    """
    Check if the model is a mixture of experts (MoE) model.
    """

    expert_weights: MutableSequence[Sequence[Tensor]]
    """
    Expert weights saved in this rank.

    The first dimension is the layer, and the second dimension is different
    parameters in the layer, e.g. up/down projection weights.
    """

    num_moe_layers: int
    """Number of MoE layers in this model."""

    num_expert_groups: int
    """Number of expert groups in this model."""

    num_logical_experts: int
    """Number of logical experts in this model."""

    num_physical_experts: int
    """Number of physical experts in this model."""

    num_local_physical_experts: int
    """Number of local physical experts in this model."""

    num_routed_experts: int
    """Number of routed experts in this model."""

    num_shared_experts: int
    """Number of shared experts in this model."""

    num_redundant_experts: int
    """Number of redundant experts in this model."""

    moe_layers: Sequence["MoERunner"]
    """List of MoE layers in this model."""

    def set_eplb_state(
        self,
        expert_load_view: Tensor,
        logical_to_physical_map: Tensor,
        logical_replica_count: Tensor,
    ) -> None:
        """
        Register the EPLB state in the MoE model.

        Since these are views of the actual EPLB state, any changes made by
        the EPLB algorithm are automatically reflected in the model's behavior
        without requiring additional method calls to set new states.

        You should also collect model's `expert_weights` here instead of in
        the weight loader, since after initial weight loading, further
        processing like quantization may be applied to the weights.

        Args:
            expert_load_view: A view of the expert load metrics tensor.
            logical_to_physical_map: Mapping from logical to physical experts.
            logical_replica_count: Count of replicas for each logical expert.
        """
        self.expert_weights = []
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None: ...


def get_mixture_of_experts_model(model: object) -> MixtureOfExperts | None:
    """Return the MixtureOfExperts contained within an arbitrary model.

    - If the model itself is a MixtureOfExperts, return the model directly.
    - If the model is a multi-modal model, and its `language_model` is a
      MixtureOfExperts, return the `language_model`.
    - If neither, return None.

    Args:
        model: Model being served.

    Returns:
        The MixtureOfExperts instance contained within the model, or None.
    """

    if is_mixture_of_experts(model):
        return model

    return None


def is_mixture_of_experts(model: object) -> TypeIs[MixtureOfExperts]:
    return (
        isinstance(model, MixtureOfExperts) and getattr(model, "num_moe_layers", 0) > 0
    )


class SupportsQuant:
    """The interface required for all models that support quantization."""

    hf_to_vllm_mapper: ClassVar["WeightsMapper | None"] = None
    packed_modules_mapping: ClassVar[dict[str, list[str]]]
    quant_config: QuantizationConfig | None = None

    def __new__(cls, *args, **kwargs) -> Self:
        instance = super().__new__(cls)

        # find config passed in arguments and attach it to model for general use
        instance.quant_config = cls._find_quant_config(*args, **kwargs)

        cls._maybe_apply_model_mapping(instance)

        return instance

    @staticmethod
    def _find_quant_config(*args, **kwargs) -> QuantizationConfig | None:
        """Find quant config passed through model constructor args"""
        from vllm.config import VllmConfig  # avoid circular import

        args_values = list(args) + list(kwargs.values())
        for arg in args_values:
            if isinstance(arg, VllmConfig):
                return arg.quant_config

            if isinstance(arg, QuantizationConfig):
                return arg

        return None

    def _maybe_apply_model_mapping(self):
        """Apply model mappings to config for proper config-model matching"""
        if self.quant_config is None:
            return
        if (hf_to_vllm_mapper := self.hf_to_vllm_mapper) is not None:
            unstacked_mapper = hf_to_vllm_mapper.get_unstacked_mapper()
            self.quant_config.apply_vllm_mapper(unstacked_mapper)
        if packed_modules_mapping := getattr(self, "packed_modules_mapping", None):
            self.quant_config.packed_modules_mapping.update(packed_modules_mapping)


@runtime_checkable
class SupportsRealtime(Protocol):
    """The interface required for all models that support transcription."""

    supports_realtime: ClassVar[Literal[True]] = True

    realtime_max_tokens: ClassVar[int] = 1
    """Maximum tokens to generate per streaming audio segment.
    Override in subclasses based on the model's expected output length."""

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: "ModelConfig",
    ) -> AsyncGenerator["PromptType", None]: ...


@overload
def supports_realtime(
    model: type[object],
) -> TypeIs[type[SupportsRealtime]]: ...


@overload
def supports_realtime(model: object) -> TypeIs[SupportsRealtime]: ...


def supports_realtime(
    model: type[object] | object,
) -> TypeIs[type[SupportsRealtime]] | TypeIs[SupportsRealtime]:
    return getattr(model, "supports_realtime", False)


@runtime_checkable
class SupportsTranscription(Protocol):
    """The interface required for all models that support transcription."""

    # Mapping from ISO639_1 language codes: language names
    supported_languages: ClassVar[Mapping[str, str]]

    supports_transcription: ClassVar[Literal[True]] = True

    supports_transcription_only: ClassVar[bool] = False
    """
    Transcription models can opt out of text generation by setting this to
    `True`.
    """
    supports_segment_timestamp: ClassVar[bool] = False
    """
    Enables the segment timestamp option for supported models by setting this to `True`.
    """

    supports_diarized_transcription: ClassVar[bool] = False
    """Enables the ``diarized_json`` response format for the model."""

    supports_explicit_language_detection: ClassVar[bool] = False
    """
    Transcription models that require an explicit language detection step
    (e.g. Whisper needs a separate forward pass to predict the language
    token) should set this to ``True`` and implement
    :meth:`get_language_detection_prompt` and
    :meth:`parse_language_detection_output` and
    :meth:`get_language_token_ids`.
    """

    no_space_languages: ClassVar[set[str]] = {"ja", "zh"}
    """
    Languages that don't need a space between words.
    For example, Japanese (ja) and Chinese (zh) don't need a space between words.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # language codes in supported_languages
        # that don't exist in the full language map
        invalid = set(cls.supported_languages) - set(LANGUAGES.keys())
        if invalid:
            raise ValueError(
                f"{cls.__name__}.supported_languages contains invalid "
                f"language codes: {sorted(invalid)}\n. "
                f"Valid choices are: {sorted(LANGUAGES.keys())}"
            )

    @classmethod
    def get_generation_prompt(
        cls,
        stt_params: Any,
    ) -> "PromptType":
        """Get the prompt for the ASR model.
        The model has control over the construction, as long as it
        returns a valid PromptType."""
        ...

    @classmethod
    def get_other_languages(cls) -> Mapping[str, str]:
        # other possible language codes from the whisper map
        return {k: v for k, v in LANGUAGES.items() if k not in cls.supported_languages}

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        """
        Ensure the language specified in the transcription request
        is a valid ISO 639-1 language code. If the request language is
        valid, but not natively supported by the model, trigger a
        warning (but not an exception).
        """
        if language is None or language in cls.supported_languages:
            return language
        elif language in cls.get_other_languages():
            logger.warning(
                "Language %r is not natively supported by %s; "
                "results may be less accurate. Supported languages: %r",
                language,
                cls.__name__,
                list(cls.supported_languages.keys()),
            )
            return language
        else:
            raise ValueError(
                f"Unsupported language: {language!r}.  Must be one of "
                f"{list(cls.supported_languages.keys())}."
            )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: "ModelConfig", task_type: Literal["transcribe", "translate"]
    ) -> Any:
        """Get the speech to text config for the ASR model."""
        ...

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: Any,
        model_config: "ModelConfig",
    ) -> int | None:
        """
        Map from audio duration to number of audio tokens produced by the ASR
        model, without running a forward pass.
        This is used for estimating the amount of processing for this audio.
        """
        return None

    @classmethod
    def post_process_output(cls, text: str) -> str:
        """
        Post-process the raw model output text.

        Some ASR models output structured formats (e.g., language tags,
        special tokens) that need to be stripped before returning to the user.

        Args:
            text: Raw decoded text from the model.

        Returns:
            Cleaned transcription text.
        """
        return text

    @classmethod
    def parse_diarized_transcript(cls, text: str) -> list[Any]:
        """Parse the model-specific diarized transcript format.

        Only models that set ``supports_diarized_transcription`` must override
        this method.
        """
        raise NotImplementedError

    @classmethod
    def get_streaming_post_processor_cls(
        cls,
    ) -> type[Any]:
        """
        Return a stateful post-processor class for streaming output deltas.

        Each instance receives the next decoded text delta and whether the
        request output is final. It returns the cleaned delta that should be
        sent to the client.
        """
        raise NotImplementedError

    @classmethod
    def get_language_detection_prompt(
        cls,
        audio: np.ndarray,
        stt_config: Any,
    ) -> "PromptType":
        """Return a prompt that triggers language detection.

        Only needs to be implemented when
        ``supports_explicit_language_detection`` is ``True``.
        """
        raise NotImplementedError

    @classmethod
    def parse_language_detection_output(
        cls,
        token_ids: list[int],
        tokenizer: object,
    ) -> str:
        """Parse the detected language from model output token IDs.

        Only needs to be implemented when
        ``supports_explicit_language_detection`` is ``True``.
        """
        raise NotImplementedError

    @classmethod
    def get_language_token_ids(
        cls,
        tokenizer: object,
    ) -> list[int] | None:
        """Return token IDs that represent valid language tokens.

        Used to constrain language detection to only produce valid language tokens.

        Only needs to be implemented when
        ``supports_explicit_language_detection`` is ``True``.
        """
        raise NotImplementedError


@overload
def supports_transcription(
    model: type[object],
) -> TypeIs[type[SupportsTranscription]]: ...


@overload
def supports_transcription(model: object) -> TypeIs[SupportsTranscription]: ...


def supports_transcription(
    model: type[object] | object,
) -> TypeIs[type[SupportsTranscription]] | TypeIs[SupportsTranscription]:
    return getattr(model, "supports_transcription", False)


@runtime_checkable
class SupportsEagleBase(Protocol):
    """Base interface for models that support EAGLE-based speculative decoding."""

    has_own_lm_head: bool = False
    """
    A flag that indicates this model has trained its own lm_head.
    """

    has_own_embed_tokens: bool = False
    """
    A flag that indicates this model has trained its own input embeddings.
    """


@overload
def supports_any_eagle(model: type[object]) -> TypeIs[type[SupportsEagleBase]]: ...


@overload
def supports_any_eagle(model: object) -> TypeIs[SupportsEagleBase]: ...


def supports_any_eagle(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagleBase]] | TypeIs[SupportsEagleBase]:
    """Check if model supports any EAGLE variant (1, 2, or 3)."""
    return supports_eagle(model) or supports_eagle3(model)


class LocalArgmaxMixin:
    """Mixin for draft model heads in speculative decoding.

    Provides a D2T-aware ``get_top_tokens`` that preserves the
    local-argmax communication reduction even when the draft vocabulary
    is smaller than the target vocabulary.

    When ``draft_id_to_target_id`` is present (shape ``(draft_vocab_size,)``,
    containing per-token offset to target vocab id), the draft argmax index
    ``k`` is mapped to the target vocab id via::

        target_id = k + draft_id_to_target_id[k]

    This is mathematically equivalent to computing the full-vocab scatter
    logits and taking the global argmax, but requires only
    O(batch * 2 * tp_size) communication instead of O(batch * vocab_size).

    Requires the subclass to expose:
        ``self.logits_processor``: LogitsProcessor
        ``self.lm_head``: ParallelLMHead
        ``self.draft_id_to_target_id`` (optional): nn.Parameter
    """

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel argmax with optional D2T remapping."""
        top = self.logits_processor.get_top_tokens(
            self.lm_head,
            hidden_states,
        )
        d2t = getattr(self, "draft_id_to_target_id", None)
        if d2t is not None:
            top = top + d2t[top]
        return top


class EagleModelMixin:
    aux_hidden_state_layers: tuple[int, ...] = ()

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_hidden_state_layers = layers

    def _maybe_add_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if layer_idx in self.aux_hidden_state_layers:
            value = hidden_states + residual if residual is not None else hidden_states
            aux_hidden_states.append(value)
        return aux_hidden_states


@runtime_checkable
class SupportsEagle(SupportsEagleBase, Protocol):
    """The interface required for models that support
    EAGLE-1 and EAGLE-2 speculative decoding."""

    supports_eagle: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-1 and EAGLE-2 
    speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """


@overload
def supports_eagle(model: type[object]) -> TypeIs[type[SupportsEagle]]: ...


@overload
def supports_eagle(model: object) -> TypeIs[SupportsEagle]: ...


def supports_eagle(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagle]] | TypeIs[SupportsEagle]:
    return isinstance(model, SupportsEagle)


@runtime_checkable
class SupportsEagle3(SupportsEagleBase, Protocol):
    """The interface required for models that support
    EAGLE-3 speculative decoding."""

    supports_eagle3: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-3 
    speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """
        Set which layers should output auxiliary hidden states for EAGLE-3.

        Args:
            layers: Tuple of layer indices that should output auxiliary
                hidden states.
        """
        parent_ref = self
        if hasattr(self, "get_language_model"):
            parent_ref = self.get_language_model()
        elif hasattr(self, "language_model"):
            parent_ref = self.language_model
        # A model that builds its decoder inside `_mark_language_model` has
        # get_language_model() return the inner decoder itself, which IS the
        # EagleModelMixin and has no further `.model`. Unwrap only when there
        # is something to unwrap.
        holder = getattr(parent_ref, "model", parent_ref)
        assert isinstance(holder, EagleModelMixin), (
            "Model instance must inherit from EagleModelMixin to set auxiliary layers"
        )
        holder._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """
        Get the default layer indices that should output auxiliary hidden states
        for EAGLE-3 for this model. Models can override this method to provide
        different default layers based on their architecture, but it is encouraged
        to instead include the layer specification in the model's config if possible.

        Returns:
            Tuple of layer indices for auxiliary hidden state outputs.
        """
        parent_ref = self
        if hasattr(self, "get_language_model"):
            parent_ref = self.get_language_model()
        elif hasattr(self, "language_model"):
            parent_ref = self.language_model
        # Same unwrap-only-if-needed rule as set_aux_hidden_state_layers.
        holder = getattr(parent_ref, "model", parent_ref)
        assert hasattr(holder, "layers"), (
            "Model instance must have 'layers' attribute to get number of layers"
        )
        num_layers = len(holder.layers)
        return (2, num_layers // 2, num_layers - 3)


@overload
def supports_eagle3(model: type[object]) -> TypeIs[type[SupportsEagle3]]: ...


@overload
def supports_eagle3(model: object) -> TypeIs[SupportsEagle3]: ...


def supports_eagle3(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagle3]] | TypeIs[SupportsEagle3]:
    return isinstance(model, SupportsEagle3)
