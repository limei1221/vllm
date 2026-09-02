# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_T = TypeVar("_T", bound=nn.Module)

logger = init_logger(__name__)

_GENERATE_SUFFIXES = [
    "ForCausalLM",
    "ForConditionalGeneration",
    "ChatModel",
    "LMHeadModel",
]


def _resolve_num_labels(hf_config: Any, text_config: Any) -> int:
    """Resolve the label count for a sequence classification head.

    ``PretrainedConfig.num_labels`` is derived from ``id2label``, which always
    carries a default of two entries. Composite configs (such as nested
    checkpoints) declare their label space on the top-level config, so reading
    ``num_labels`` from ``get_text_config()`` silently returns that default and
    builds a score head of the wrong size.

    Prefer the top-level config whenever it declares a label space of its own,
    mirroring the ``classifier_from_token`` / ``method`` lookups in
    ``as_seq_cls_model``.
    """
    if text_config is hf_config:
        return hf_config.num_labels

    from transformers import PretrainedConfig

    if hf_config.num_labels != PretrainedConfig().num_labels:
        return hf_config.num_labels
    return text_config.num_labels


class SequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        hf_config = vllm_config.model_config.hf_config
        text_config = hf_config.get_text_config()
        method = getattr(hf_config, "method", getattr(text_config, "method", None))
        tokens = getattr(
            hf_config,
            "classifier_from_token",
            getattr(text_config, "classifier_from_token", None),
        )

        if method is None:
            return

        assert tokens is not None
        assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"

        if method == "from_2_way_softmax":
            assert len(tokens) == 2
            hf_config.num_labels = 1
            text_config.num_labels = 1
        else:
            hf_config.num_labels = len(tokens)
            text_config.num_labels = len(tokens)

        # `llm as reranker` defaults to not using separating token.
        use_sep_token = getattr(text_config, "use_sep_token", False)
        text_config.use_sep_token = use_sep_token


def _get_language_model_for_seq_cls(model: nn.Module) -> nn.Module:
    """
    Get the language model component for sequence classification conversion.
    For VLMs, returns the inner language model. For standard LLMs, returns model itself.
    """
    for attr_name in ("language_model", "lm", "text_model"):
        if hasattr(model, attr_name):
            candidate = getattr(model, attr_name)
            if (
                isinstance(candidate, nn.Module)
                and candidate is not model
                and hasattr(candidate, "model")
            ):
                return candidate

    for name, child in model.named_children():
        child_name = type(child).__name__
        if ("ForCausalLM" in child_name or "LMHead" in child_name) and hasattr(
            child, "model"
        ):
            return child

    return model


@contextmanager
def _disable_seq_cls_loading_on_inner_model(language_model, is_vlm: bool):
    """
    Context manager to temporarily disable sequence classification loading
    on inner VLM models to prevent recursive seq_cls_model_loader calls.
    """
    if not is_vlm:
        yield
        return

    inner_hf_config = getattr(language_model, "config", None)
    if inner_hf_config is None:
        yield
        return

    inner_text_config = inner_hf_config.get_text_config()
    original_method = getattr(inner_text_config, "method", None)
    original_tokens = getattr(inner_text_config, "classifier_from_token", None)
    original_hf_tokens = getattr(inner_hf_config, "classifier_from_token", None)

    try:
        if original_method is not None:
            inner_text_config.method = None
        if original_tokens is not None:
            inner_text_config.classifier_from_token = None
        if original_hf_tokens is not None:
            inner_hf_config.classifier_from_token = None
        yield
    finally:
        if original_method is not None:
            inner_text_config.method = original_method
        if original_tokens is not None:
            inner_text_config.classifier_from_token = original_tokens
        if original_hf_tokens is not None:
            inner_hf_config.classifier_from_token = original_hf_tokens


def load_weights_using_from_2_way_softmax(
    model, weights: Iterable[tuple[str, torch.Tensor]]
):
    # refer to https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    model_config = model.vllm_config.model_config
    hf_config = model.config
    text_config = hf_config.get_text_config()

    tokens: list[str] = getattr(
        hf_config,
        "classifier_from_token",
        getattr(text_config, "classifier_from_token", []),
    )
    assert len(tokens) == 2

    language_model = _get_language_model_for_seq_cls(model)
    is_vlm = language_model is not model
    using_vlm_head = is_vlm and hasattr(language_model, "score")

    language_model.lm_head = ParallelLMHead(
        text_config.vocab_size,
        text_config.hidden_size,
    )
    if text_config.tie_word_embeddings:
        # embed_tokens is the assumed name for input embeddings. If the model does not
        # have this attribute, we fall back to get_input_embeddings(), which is used by
        # the Transformers modeling backend.
        text_backbone = language_model.model
        embed_tokens = (
            text_backbone.embed_tokens
            if hasattr(text_backbone, "embed_tokens")
            else text_backbone.get_input_embeddings()
        )
        language_model.lm_head = language_model.lm_head.tie_weights(embed_tokens)

    with _disable_seq_cls_loading_on_inner_model(language_model, is_vlm):
        loaded_weights = model._load_pooling_model_weights(weights)

    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
    )

    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])
    lm_head_weight = language_model.lm_head.weight
    score_weight = lm_head_weight.data[[true_id]].to(
        torch.float32
    ) - lm_head_weight.data[[false_id]].to(torch.float32)

    score_layer = language_model.score if using_vlm_head else model.score
    param = score_layer.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, score_weight)

    del language_model.lm_head

    score_weight_name = (
        "language_model.score.weight" if using_vlm_head else "score.weight"
    )
    loaded_weights.add(score_weight_name)

    lm_head_name = "lm_head.weight"
    if hf_to_vllm_mapper := getattr(model, "hf_to_vllm_mapper", None):
        lm_head_name = hf_to_vllm_mapper._map_name(lm_head_name)
    loaded_weights.discard(lm_head_name)
    return loaded_weights


def load_weights_no_post_processing(model, weights: Iterable[tuple[str, torch.Tensor]]):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    model_config = model.vllm_config.model_config
    text_config = model.config.get_text_config()

    tokens: list[str] = getattr(text_config, "classifier_from_token", [])
    assert len(tokens) > 0

    language_model = _get_language_model_for_seq_cls(model)
    is_vlm = language_model is not model
    using_vlm_head = is_vlm and hasattr(language_model, "score")

    language_model.lm_head = ParallelLMHead(
        text_config.vocab_size,
        text_config.hidden_size,
    )
    if text_config.tie_word_embeddings:
        # embed_tokens is the assumed name for input embeddings. If the model does not
        # have this attribute, we fall back to get_input_embeddings(), which is used by
        # the Transformers modeling backend.
        text_backbone = language_model.model
        embed_tokens = (
            text_backbone.embed_tokens
            if hasattr(text_backbone, "embed_tokens")
            else text_backbone.get_input_embeddings()
        )
        language_model.lm_head = language_model.lm_head.tie_weights(embed_tokens)

    with _disable_seq_cls_loading_on_inner_model(language_model, is_vlm):
        # Bypass ModelForSequenceClassification to avoid infinite recursion
        loaded_weights = model._load_pooling_model_weights(weights)

    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
    )

    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    score_weight = language_model.lm_head.weight.data[token_ids]

    score_layer = language_model.score if using_vlm_head else model.score
    param = score_layer.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, score_weight)

    del language_model.lm_head

    score_weight_name = (
        "language_model.score.weight" if using_vlm_head else "score.weight"
    )
    loaded_weights.add(score_weight_name)

    lm_head_name = "lm_head.weight"
    if hf_to_vllm_mapper := getattr(model, "hf_to_vllm_mapper", None):
        lm_head_name = hf_to_vllm_mapper._map_name(lm_head_name)
    loaded_weights.discard(lm_head_name)
    return loaded_weights


SEQ_CLS_LOAD_METHODS = {
    "from_2_way_softmax": load_weights_using_from_2_way_softmax,
    "no_post_processing": load_weights_no_post_processing,
}


def seq_cls_model_loader(model, weights: Iterable[tuple[str, torch.Tensor]]):
    # Online convert ForCausalLM into ForSequenceClassification model.
    # - from_2_way_softmax:
    #   - Qwen3ForCausalLM
    #     - Qwen3-Reranker
    #   - Qwen2ForCausalLM
    #     - mxbai-rerank-v2
    # - no_post_processing:
    #   - GemmaForCausalLM
    #     - bge-reranker-v2-gemma

    hf_config = model.vllm_config.model_config.hf_config
    text_config = hf_config.get_text_config()
    method = getattr(hf_config, "method", getattr(text_config, "method", None))
    assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"
    return SEQ_CLS_LOAD_METHODS[method](model, weights)
