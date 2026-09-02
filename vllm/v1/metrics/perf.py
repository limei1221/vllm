# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Analytic flops/memory estimation module for transformer components,
to help derive MFU (Model Flops Utilization) stats for a running model.
"""

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import prometheus_client
import torch
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Self

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_dtype_size,
    get_kv_cache_torch_dtype,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


class InvalidComponent(Exception):
    """
    Custom exception to indicate that a certain ComponentMetric is not
    applicable to the given VllmConfig.
    """

    pass


# Mapping from quantization method name to effective weight byte size.
# Used by both AttentionQuantizationConfigParser and
# FfnQuantizationConfigParser to determine the weight_byte_size for
# flops/memory estimation.
#
# NOTE: Methods like GPTQ support variable bit-widths
# (e.g., 4-bit and 8-bit). We default to 4-bit (0.5 bytes) since this
# is by far the most common configuration.
_QUANT_WEIGHT_BYTE_SIZE: dict[str, float] = {
    # FP8 methods (1 byte per weight)
    "fp8": 1,
    "fbgemm_fp8": 1,
    "ptpc_fp8": 1,
    "fp_quant": 1,
    "modelopt": 1,
    "modelopt_mxfp8": 1,
    # FP4 / INT4 methods (0.5 bytes per weight)
    "mxfp4": 0.5,
    "awq": 0.5,
    "awq_marlin": 0.5,
    "gptq": 0.5,
    "gptq_marlin": 0.5,
    "modelopt_fp4": 0.5,
    "petit_nvfp4": 0.5,
    "compressed-tensors": 0.5,
    "torchao": 0.5,
    "quark": 0.5,
    "moe_wna16": 0.5,
    "inc": 0.5,
    "experts_int8": 1,
}


#### Basic Data Types ####


@dataclass
class DebugPerfStats:
    ## Stats for debugging the metrics calculation
    calc_duration: float = 0.0  # time spent calculating these stats
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    context_breakdown: dict[str, int] | None = None
    num_flops_per_gpu_breakdown: dict[str, int] | None = None
    num_read_bytes_per_gpu_breakdown: dict[str, int] | None = None
    num_write_bytes_per_gpu_breakdown: dict[str, int] | None = None


@dataclass
class PerfStats:
    num_flops_per_gpu: int = 0
    num_read_bytes_per_gpu: int = 0
    num_write_bytes_per_gpu: int = 0
    debug_stats: DebugPerfStats | None = None


@dataclass
class ExecutionContext:
    """
    Represents an execution context for a batch of requests.

    This class aggregates statistics across multiple requests in a batch,
    separately tracking prefill and decode phases.

    Example)
    - Batch with one full prefill (2048 tokens) and one decode (1 token, 8192 context):
      ctx = ExecutionContext()
      ctx.add(2048, 2048, is_prefill=True)
      ctx.add(1, 8192, is_prefill=False)
    """

    # Prefill phase statistics
    num_prefill_requests: int = 0
    prefill_num_tokens: int = 0  # sum of num_tokens for prefill requests
    prefill_context_len: int = 0  # sum of context_len for prefill requests
    prefill_token_context_product: int = 0  # sum of (num_tokens * context_len)

    # Decode phase statistics
    num_decode_requests: int = 0
    decode_num_tokens: int = 0  # sum of num_tokens for decode requests
    decode_context_len: int = 0  # sum of context_len for decode requests
    decode_token_context_product: int = 0  # sum of (num_tokens * context_len)

    def add(self, num_tokens: int, context_len: int, is_prefill: bool) -> None:
        """Add a single request's statistics to this batch context."""
        if is_prefill:
            self.num_prefill_requests += 1
            self.prefill_num_tokens += num_tokens
            self.prefill_context_len += context_len
            self.prefill_token_context_product += num_tokens * context_len
        else:
            self.num_decode_requests += 1
            self.decode_num_tokens += num_tokens
            self.decode_context_len += context_len
            self.decode_token_context_product += num_tokens * context_len

    def total_num_tokens(self) -> int:
        """Total number of tokens across all requests in the batch."""
        return self.prefill_num_tokens + self.decode_num_tokens

    def total_token_context_product(self) -> int:
        """Total sum of (num_tokens * context_len) across all requests."""
        return self.prefill_token_context_product + self.decode_token_context_product

    def num_logits_tokens(self) -> int:
        """Number of tokens that require logits computation (unembedding).

        For prefill, only the last token per request needs logits.
        For decode, all tokens need logits.
        """
        return self.num_prefill_requests + self.decode_num_tokens

    @classmethod
    def from_single_request(
        cls, num_tokens: int, context_len: int, is_prefill: bool
    ) -> "ExecutionContext":
        """Create an ExecutionContext from a single request.

        This is a convenience method primarily for testing.
        """
        ctx = cls()
        ctx.add(num_tokens, context_len, is_prefill)
        return ctx


class ParsedArgs:
    """
    Syntactic sugar so that Parsers can use dot notations
    to access/update the parsed arguments.

    e.g.)
        args = ParsedArgs()
        args.x = 3
        args.y = args.x + 1
    """

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def model_dump(self) -> dict[str, Any]:
        return vars(self).copy()


#### Abstract ####


class Parser(Protocol):
    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """
        Parse the vllm config and update the current ParsedArgs and pass it on.
        If the parser isn't applicable to the vllm_config, it will do nothing.
        """
        ...


class ParserChain:
    """
    Applies chain of parser in a sequential order.
    Later parsers might overwrite results from previous parsers,
    so parsers should be chained in the appropriate order if they
    are not mutually exclusive.
    """

    def __init__(self, *parsers: Parser) -> None:
        self.parsers = list(parsers)

    def add_parser(self, parser: Parser) -> None:
        self.parsers.append(parser)

    def parse(self, vllm_config: VllmConfig) -> ParsedArgs:
        args = ParsedArgs()
        for parser in self.parsers:
            args = parser.parse(args, vllm_config)
        return args


_COMPONENT_METRICS_REGISTRY: dict[str, type["ComponentMetrics"]] = {}


class ComponentMetrics(BaseModel, ABC):
    """
    Each concrete ComponentMetrics class is associated with:
    - fields that are required for metric derivation
      (fields are specified/validated through pydantic model)
    - parser to parse VllmConfig into fields
    - metric methods that derive flops/bytes for a given execution context
    """

    @classmethod
    @abstractmethod
    def component_type(cls) -> str: ...

    @classmethod
    @abstractmethod
    def get_parser(cls) -> ParserChain:
        """
        Return a ParserChain that provides values for all required fields.
        The returned parser chain must populate ParsedArgs with values for every
        field defined on this ComponentMetrics class. Missing fields will cause
        a ValidationError when from_vllm_config() is called.
        See individual Parser docstrings for which args they provide, and field
        comments on ComponentMetrics subclasses for which parser provides each field.
        """
        ...

    def __init_subclass__(cls):
        _COMPONENT_METRICS_REGISTRY[cls.component_type()] = cls

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> Self:
        """
        Instantiate this class from VllmConfig.
        Raises ValidationError if parsing fails.
        """

        parser = cls.get_parser()
        parsed_args = parser.parse(vllm_config)
        try:
            return cls.model_validate(parsed_args.model_dump())
        except ValidationError as e:
            raise InvalidComponent(f"Invalid {cls.component_type()} config: {e}") from e

    @classmethod
    def registered_metrics(cls) -> Iterable[type["ComponentMetrics"]]:
        return iter(_COMPONENT_METRICS_REGISTRY.values())

    @abstractmethod
    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    @abstractmethod
    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    @abstractmethod
    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    def get_num_flops(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(self.get_num_flops_breakdown(ctx, per_gpu).values())

    def get_read_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(self.get_read_bytes_breakdown(ctx, per_gpu).values())

    def get_write_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(self.get_write_bytes_breakdown(ctx, per_gpu).values())


#### parsers ####


class BaseConfigParser(Parser):
    """
    Parses base model configuration.
    Provides: vocab_size, hidden_size, num_attention_heads, num_hidden_layers,
    weight_byte_size, activation_byte_size, dp_size, tp_size, pp_size, enable_ep
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        model_config = vllm_config.model_config

        args.vocab_size = model_config.get_vocab_size()
        args.hidden_size = model_config.get_hidden_size()
        # NOTE: model_config.get_attention_heads() divide by TP
        # so we access field manually here to get total num_heads
        args.num_attention_heads = get_required(
            model_config.hf_text_config, "num_attention_heads"
        )
        args.num_hidden_layers = get_required(
            model_config.hf_text_config, "num_hidden_layers"
        )

        model_dtype = vllm_config.model_config.dtype

        if isinstance(model_dtype, torch.dtype):
            torch_dtype = model_dtype
        elif isinstance(model_dtype, str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
        else:
            # FIXME: handle this better
            logger.warning(
                "Unknown model_dtype %s, defaulting to bfloat16",
                model_dtype,
            )
            torch_dtype = torch.bfloat16

        args.weight_byte_size = get_dtype_size(torch_dtype)

        # FIXME: handle this better by parsing whether activations use
        # bf16, fp32, etc...
        args.activation_byte_size = 2

        args.dp_size = vllm_config.parallel_config.data_parallel_size
        args.tp_size = vllm_config.parallel_config.tensor_parallel_size
        args.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        args.enable_ep = vllm_config.parallel_config.enable_expert_parallel

        return args


#### Attention ####


class BaseAttentionConfigParser(Parser):
    """
    Parses attention-specific configuration.
    Provides: num_key_value_heads, head_dim, cache_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        model_config = vllm_config.model_config

        args.num_key_value_heads = model_config.get_total_num_kv_heads()
        args.head_dim = model_config.get_head_size()

        model_dtype = vllm_config.model_config.dtype
        cache_dtype = vllm_config.cache_config.cache_dtype

        kv_cache_torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
        args.cache_byte_size = get_dtype_size(kv_cache_torch_dtype)

        return args


class AttentionQuantizationConfigParser(Parser):
    """
    Parses quantization configuration for attention layers.
    Overrides: weight_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()
        if quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            raise InvalidComponent(
                f"Unsupported quantization method for attention metrics: {quant_method}"
            )

        return args


class AttentionDetectionParser(Parser):
    """
    Prevents standard AttentionMetrics from being instantiated for MLA models.
    MLA models should use MLAAttentionMetrics instead.
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        if vllm_config.model_config.is_deepseek_mla:
            raise InvalidComponent(
                "Model uses MLA attention; use MLAAttentionMetrics instead"
            )
        return args


class AttentionMetrics(ComponentMetrics):
    # From BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_attention_heads: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    tp_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # From BaseAttentionConfigParser
    num_key_value_heads: int = Field(..., gt=0)
    head_dim: int = Field(..., gt=0)
    cache_byte_size: int = Field(..., gt=0)

    # From BaseConfig Parser, overridden by AttentionQuantizationConfigParser
    weight_byte_size: int | float = Field(..., gt=0)

    # TODO: discern cases where we have mixture of different attention layer types
    # such as SWA, MLA, etc.

    @classmethod
    def component_type(cls) -> str:
        return "attn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            AttentionDetectionParser(),
            BaseConfigParser(),
            BaseAttentionConfigParser(),
            AttentionQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()
        TC = ctx.total_token_context_product()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        return {
            "qkv_proj": 2 * T * D * (q + 2 * kv) * d * L,
            "attn_qk": 2 * q * TC * d * L,
            "attn_av": 2 * q * TC * d * L,
            "out_proj": 2 * T * D * q * d * L,
        }

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        read_bytes = {}

        read_bytes["qkv_input"] = T * D * self.activation_byte_size * L
        read_bytes["qkv_weight"] = int(D * (q + 2 * kv) * d * self.weight_byte_size * L)

        # Attention input reads differ between prefill and decode
        # Prefill: read Q, K, V activations (all in activation_byte_size)
        if ctx.prefill_num_tokens > 0:
            read_bytes["attn_input"] = (
                (ctx.prefill_num_tokens * q + 2 * ctx.prefill_context_len * kv)
                * d
                * self.activation_byte_size
                * L
            )

        # Decode: read Q activations + read K, V from cache (in cache_byte_size)
        if ctx.decode_num_tokens > 0:
            read_bytes["attn_input"] = read_bytes.get("attn_input", 0) + (
                ctx.decode_num_tokens * q * d * self.activation_byte_size * L
                + 2 * ctx.decode_context_len * kv * d * self.cache_byte_size * L
            )

        read_bytes["out_input"] = T * q * d * self.activation_byte_size * L
        read_bytes["out_weight"] = int(q * d * D * self.weight_byte_size * L)

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for attention layers."""
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        return {
            "qkv_output": T * (q + 2 * kv) * d * self.activation_byte_size * L,
            "kv_cache": 2 * T * kv * d * self.cache_byte_size * L,
            "out_output": T * D * self.activation_byte_size * L,
        }


#### MLA Attention ####


class MLADetectionParser(Parser):
    """
    Validates that the model uses MLA attention.
    Raises InvalidComponent if the model does not use MLA,
    so MLAAttentionMetrics is silently skipped for non-MLA models.
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        if not vllm_config.model_config.is_deepseek_mla:
            raise InvalidComponent("Model does not use MLA attention")
        return args


class MLAConfigParser(Parser):
    """
    Parses MLA-specific configuration fields.
    Provides: kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
    v_head_dim, q_lora_rank
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        model_config = vllm_config.model_config
        cfg = model_config.hf_text_config

        args.kv_lora_rank = get_required(cfg, "kv_lora_rank")
        args.qk_nope_head_dim = get_required(cfg, "qk_nope_head_dim")
        args.qk_rope_head_dim = get_required(cfg, "qk_rope_head_dim")
        args.v_head_dim = get_required(cfg, "v_head_dim")
        args.q_lora_rank = getattr(cfg, "q_lora_rank", None)

        model_dtype = vllm_config.model_config.dtype
        cache_dtype = vllm_config.cache_config.cache_dtype
        kv_cache_torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
        args.cache_byte_size = get_dtype_size(kv_cache_torch_dtype)

        return args


class MLAAttentionMetrics(ComponentMetrics):
    """
    Performance metrics for Multi-Latent Attention (MLA) layers.

    MLA uses a compressed latent representation for KV cache:
    - KV cache stores a single compressed vector of size
      (kv_lora_rank + qk_rope_head_dim) per token per layer,
      instead of 2 * num_kv_heads * head_dim as in standard MHA/GQA.
    - Q path uses optional low-rank compression:
      h -> q_lora_rank -> num_heads * qk_head_dim
    - KV path: h -> (kv_lora_rank + qk_rope_head_dim),
      then kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)

    Used by DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, and similar models.
    """

    # From BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_attention_heads: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    tp_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # From BaseConfigParser, can be overridden by AttentionQuantizationConfigParser
    weight_byte_size: int | float = Field(..., gt=0)

    # From MLAConfigParser
    kv_lora_rank: int = Field(..., gt=0)
    qk_nope_head_dim: int = Field(..., gt=0)
    qk_rope_head_dim: int = Field(..., gt=0)
    v_head_dim: int = Field(..., gt=0)
    q_lora_rank: int | None = Field(None)
    cache_byte_size: int = Field(..., gt=0)

    @classmethod
    def component_type(cls) -> str:
        return "mla_attn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            MLADetectionParser(),
            BaseConfigParser(),
            MLAConfigParser(),
            AttentionQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate flops breakdown for MLA attention layers.

        MLA projection structure:
        - Q path: h -> q_lora_rank -> num_heads * qk_head_dim
          (or h -> num_heads * qk_head_dim if q_lora_rank is None)
        - KV path: h -> (kv_lora_rank + qk_rope_head_dim),
          then kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
        - Attention: Q @ K^T and attn @ V
        - Output: num_heads * v_head_dim -> h
        """
        L = self.num_hidden_layers
        D = self.hidden_size
        q = self.num_attention_heads
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_d = self.v_head_dim
        c = self.kv_lora_rank
        r = self.qk_rope_head_dim
        q_rank = self.q_lora_rank

        T = ctx.total_num_tokens()
        TC = ctx.total_token_context_product()

        if per_gpu:
            L //= self.pp_size
            q = max(1, q // self.tp_size)

        flops: dict[str, int] = {}

        # Q projection
        if q_rank is not None:
            # Two-stage: h -> q_lora_rank -> num_heads * qk_head_dim
            flops["q_a_proj"] = 2 * T * D * q_rank * L
            flops["q_b_proj"] = 2 * T * q_rank * q * qk_head_dim * L
        else:
            # Direct: h -> num_heads * qk_head_dim
            flops["q_proj"] = 2 * T * D * q * qk_head_dim * L

        # KV projection (always compressed, shared across heads)
        # kv_a: h -> (kv_lora_rank + qk_rope_head_dim)  [replicated]
        flops["kv_a_proj"] = 2 * T * D * (c + r) * L
        # kv_b: kv_lora_rank -> num_heads * (qk_nope + v_head_dim)
        flops["kv_b_proj"] = 2 * T * c * q * (self.qk_nope_head_dim + v_d) * L

        # Attention core
        flops["attn_qk"] = 2 * q * TC * qk_head_dim * L
        flops["attn_av"] = 2 * q * TC * v_d * L

        # Output projection: num_heads * v_head_dim -> h
        flops["out_proj"] = 2 * T * q * v_d * D * L

        return flops

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate read memory traffic for MLA attention layers."""
        L = self.num_hidden_layers
        D = self.hidden_size
        q = self.num_attention_heads
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_d = self.v_head_dim
        c = self.kv_lora_rank
        r = self.qk_rope_head_dim
        q_rank = self.q_lora_rank

        T = ctx.total_num_tokens()
        # Compressed KV cache size per token
        kv_compressed_dim = c + r

        if per_gpu:
            L //= self.pp_size
            q = max(1, q // self.tp_size)

        read_bytes: dict[str, int] = {}

        # Q projection weight + input reads
        if q_rank is not None:
            read_bytes["q_a_input"] = T * D * self.activation_byte_size * L
            read_bytes["q_a_weight"] = int(D * q_rank * self.weight_byte_size * L)
            read_bytes["q_b_input"] = T * q_rank * self.activation_byte_size * L
            read_bytes["q_b_weight"] = int(
                q_rank * q * qk_head_dim * self.weight_byte_size * L
            )
        else:
            read_bytes["q_input"] = T * D * self.activation_byte_size * L
            read_bytes["q_weight"] = int(
                D * q * qk_head_dim * self.weight_byte_size * L
            )

        # KV projection weight + input reads
        # kv_a is replicated (not TP-sharded)
        read_bytes["kv_a_input"] = T * D * self.activation_byte_size * L
        read_bytes["kv_a_weight"] = int(
            D * kv_compressed_dim * self.weight_byte_size * L
        )
        # kv_b is TP-sharded along heads
        read_bytes["kv_b_input"] = T * c * self.activation_byte_size * L
        read_bytes["kv_b_weight"] = int(
            c * q * (self.qk_nope_head_dim + v_d) * self.weight_byte_size * L
        )

        # Attention input reads
        # Prefill: read Q activations + K,V from kv_b_proj output
        if ctx.prefill_num_tokens > 0:
            read_bytes["attn_input"] = (
                ctx.prefill_num_tokens * q * qk_head_dim * self.activation_byte_size * L
                + ctx.prefill_context_len
                * q
                * (qk_head_dim + v_d)
                * self.activation_byte_size
                * L
            )

        # Decode: read Q activations + read compressed KV from cache
        if ctx.decode_num_tokens > 0:
            read_bytes["attn_input"] = read_bytes.get("attn_input", 0) + (
                ctx.decode_num_tokens * q * qk_head_dim * self.activation_byte_size * L
                + ctx.decode_context_len * kv_compressed_dim * self.cache_byte_size * L
            )

        # Output projection reads
        read_bytes["out_input"] = T * q * v_d * self.activation_byte_size * L
        read_bytes["out_weight"] = int(q * v_d * D * self.weight_byte_size * L)

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for MLA attention layers."""
        L = self.num_hidden_layers
        D = self.hidden_size
        q = self.num_attention_heads
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_d = self.v_head_dim
        c = self.kv_lora_rank
        r = self.qk_rope_head_dim
        q_rank = self.q_lora_rank

        T = ctx.total_num_tokens()
        kv_compressed_dim = c + r

        if per_gpu:
            L //= self.pp_size
            q = max(1, q // self.tp_size)

        write_bytes: dict[str, int] = {}

        # Q projection outputs
        if q_rank is not None:
            write_bytes["q_a_output"] = T * q_rank * self.activation_byte_size * L
            write_bytes["q_b_output"] = (
                T * q * qk_head_dim * self.activation_byte_size * L
            )
        else:
            write_bytes["q_output"] = (
                T * q * qk_head_dim * self.activation_byte_size * L
            )

        # KV projection outputs
        write_bytes["kv_a_output"] = (
            T * kv_compressed_dim * self.activation_byte_size * L
        )
        write_bytes["kv_b_output"] = (
            T * q * (self.qk_nope_head_dim + v_d) * self.activation_byte_size * L
        )

        # KV cache write: one compressed vector per token
        # (kv_lora_rank + qk_rope_head_dim) instead of
        # 2 * num_kv_heads * head_dim in standard MHA
        write_bytes["kv_cache"] = T * kv_compressed_dim * self.cache_byte_size * L

        # Output projection
        write_bytes["out_output"] = T * D * self.activation_byte_size * L

        return write_bytes


#### Ffn ####


class FfnQuantizationConfigParser(Parser):
    """
    Parses quantization configuration for FFN layers.

    Overrides: weight_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()
        if quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            raise InvalidComponent(
                f"Unsupported quantization method for FFN metrics: {quant_method}"
            )

        return args


#### Unembed ####


#### ModelMetrics ####


class ModelMetrics:
    def __init__(self, vllm_config: VllmConfig) -> None:
        """
        Parse vllm_config to instantiate metrics for each component.
        is_enabled() will return False if no component metrics could be instantiated.
        """

        self.vllm_config = vllm_config

        self.metrics: list[ComponentMetrics] = []
        for metric_cls in ComponentMetrics.registered_metrics():
            try:
                metric = metric_cls.from_vllm_config(vllm_config)
                self.metrics.append(metric)
                logger.info(
                    "Instantiated ComponentMetrics [%s] with (%s)",
                    metric.component_type(),
                    str(metric),
                )
            except InvalidComponent as e:
                logger.debug(
                    "Failed to instantiate %s from %s",
                    metric_cls.component_type(),
                    str(e),
                )

    def is_enabled(self) -> bool:
        return len(self.metrics) > 0

    def get_num_flops(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(metric.get_num_flops(ctx, per_gpu) for metric in self.metrics)

    def get_read_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(metric.get_read_bytes(ctx, per_gpu) for metric in self.metrics)

    def get_write_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(metric.get_write_bytes(ctx, per_gpu) for metric in self.metrics)

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_num_flops_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_read_bytes_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_write_bytes_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_step_perf_stats_per_gpu(
        self, scheduler_output: SchedulerOutput
    ) -> PerfStats:
        """
        Calculate perf stats for the current step based on scheduled tokens.
        """

        t0 = time.monotonic()

        # Build a single batch context
        ctx = ExecutionContext()

        # Process new requests (these are in prefill phase)
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            # For new requests, context_len = num_computed_tokens + num_tokens
            # num_computed_tokens represents previously computed tokens in the sequence
            context_len = new_req.num_computed_tokens + num_tokens
            ctx.add(num_tokens, context_len, is_prefill=True)

        # Process cached requests (continuing requests)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            # For cached requests, we have the current num_computed_tokens
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            context_len = num_computed_tokens + num_tokens

            # Cached requests are typically in decode phase (num_tokens == 1)
            # unless they're doing chunked prefill (num_tokens > 1)
            is_prefill = num_tokens > 1
            ctx.add(num_tokens, context_len, is_prefill)

        num_flops_breakdown = self.get_num_flops_breakdown(ctx, True)
        read_bytes_breakdown = self.get_read_bytes_breakdown(ctx, True)
        write_bytes_breakdown = self.get_write_bytes_breakdown(ctx, True)
        perf_stats = PerfStats(
            sum(num_flops_breakdown.values()),
            sum(read_bytes_breakdown.values()),
            sum(write_bytes_breakdown.values()),
        )

        if envs.VLLM_DEBUG_MFU_METRICS:
            perf_stats.debug_stats = DebugPerfStats(
                time.monotonic() - t0,
                ctx.num_prefill_requests,
                ctx.num_decode_requests,
                asdict(ctx),
                num_flops_breakdown,
                read_bytes_breakdown,
                write_bytes_breakdown,
            )

        return perf_stats


#### Logging ####


class PerfMetricsDebugLogging:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_calc_duration: float = 0.0
        self.total_num_prefill_requests: int = 0
        self.total_num_decode_requests: int = 0
        self.total_num_batches: int = 0
        self.total_context_breakdown: dict[str, int] = {}
        self.total_num_flops_per_gpu_breakdown: dict[str, int] = {}
        self.total_read_bytes_per_gpu_breakdown: dict[str, int] = {}
        self.total_write_bytes_per_gpu_breakdown: dict[str, int] = {}

    def observe(self, debug_stats: DebugPerfStats) -> None:
        self.total_calc_duration += debug_stats.calc_duration
        self.total_num_prefill_requests += debug_stats.num_prefill_requests
        self.total_num_decode_requests += debug_stats.num_decode_requests
        self.total_num_batches += 1

        for dst, src in zip(
            [
                self.total_context_breakdown,
                self.total_num_flops_per_gpu_breakdown,
                self.total_read_bytes_per_gpu_breakdown,
                self.total_write_bytes_per_gpu_breakdown,
            ],
            [
                debug_stats.context_breakdown,
                debug_stats.num_flops_per_gpu_breakdown,
                debug_stats.num_read_bytes_per_gpu_breakdown,
                debug_stats.num_write_bytes_per_gpu_breakdown,
            ],
        ):
            assert isinstance(src, dict)
            for key, val in src.items():
                dst[key] = dst.get(key, 0) + val

    def log(self, log_fn, log_prefix: str, delta_time: float):
        # pretty print breakdowns
        total_num_flops_per_gpu_breakdown = {
            k: f"{v / 1e12:.1f}TF"
            for k, v in self.total_num_flops_per_gpu_breakdown.items()
        }
        total_read_bytes_per_gpu_breakdown = {
            k: f"{v / 1e9:.1f}GB"
            for k, v in self.total_read_bytes_per_gpu_breakdown.items()
        }
        total_write_bytes_per_gpu_breakdown = {
            k: f"{v / 1e9:.1f}GB"
            for k, v in self.total_write_bytes_per_gpu_breakdown.items()
        }

        logger.debug(
            "%sMFU details: %s",
            log_prefix,
            json.dumps(
                {
                    "prefill_reqs": self.total_num_prefill_requests,
                    "decode_reqs": self.total_num_decode_requests,
                    "num_batches": self.total_num_batches,
                    "context_breakdown": self.total_context_breakdown,
                    "flops_breakdown": total_num_flops_per_gpu_breakdown,
                    "num_read_bytes_breakdown": total_read_bytes_per_gpu_breakdown,
                    "num_write_bytes_breakdown": (total_write_bytes_per_gpu_breakdown),
                    "duration": f"{delta_time:.1f}s",
                    "mfu_calc_overhead": (
                        f"{self.total_calc_duration / delta_time:.1%}"
                    ),
                },
                indent=2,
            ),
        )


class PerfMetricsLogging:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size

        self.debug_logging: PerfMetricsDebugLogging | None = None
        if envs.VLLM_DEBUG_MFU_METRICS:
            self.debug_logging = PerfMetricsDebugLogging()

        self.reset()

    def reset(self):
        self.last_log_time = time.monotonic()

        self.total_num_flops_per_gpu: int = 0
        self.total_read_bytes_per_gpu: int = 0
        self.total_write_bytes_per_gpu: int = 0

        if self.debug_logging:
            self.debug_logging.reset()

    def observe(self, perf_stats: PerfStats) -> None:
        self.total_num_flops_per_gpu += perf_stats.num_flops_per_gpu
        self.total_read_bytes_per_gpu += perf_stats.num_read_bytes_per_gpu
        self.total_write_bytes_per_gpu += perf_stats.num_write_bytes_per_gpu

        if self.debug_logging:
            assert perf_stats.debug_stats is not None
            self.debug_logging.observe(perf_stats.debug_stats)

    def log(self, log_fn=logger.info, log_prefix: str = "") -> None:
        if not (
            self.total_num_flops_per_gpu
            or self.total_read_bytes_per_gpu
            or self.total_write_bytes_per_gpu
        ):
            return

        now = time.monotonic()
        delta_time = now - self.last_log_time

        if delta_time <= 0.0:
            avg_tflops_per_gpu = 0.0
            avg_gbps_per_gpu = 0.0
        else:
            avg_tflops_per_gpu = self.total_num_flops_per_gpu / delta_time / 1e12
            avg_gbps_per_gpu = (
                (self.total_read_bytes_per_gpu + self.total_write_bytes_per_gpu)
                / delta_time
                / 1e9
            )

        log_fn(
            "%sMFU: %.1f TF/s/GPU %.1f GB/s/GPU",
            log_prefix,
            avg_tflops_per_gpu,
            avg_gbps_per_gpu,
        )

        if self.debug_logging:
            self.debug_logging.log(log_fn, log_prefix, delta_time)

        self.reset()


#### Prometheus Integration ####


class PerfMetricsProm:
    """Record performance metrics in Prometheus.

    Average TFLOPS (tera floating-point operations per second) can be
    calculated using a PromQL query:

      rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

    Average memory bandwidth in GB/s can be calculated using:

      (rate(vllm:estimated_read_bytes_per_gpu_total[1m]) +
       rate(vllm:estimated_write_bytes_per_gpu_total[1m])) / 1e9
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        counter_flops = self._counter_cls(
            name="vllm:estimated_flops_per_gpu_total",
            documentation=(
                "Estimated number of floating point operations per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_flops = create_metric_per_engine(
            counter_flops, per_engine_labelvalues
        )

        counter_read_bytes = self._counter_cls(
            name="vllm:estimated_read_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes read from memory per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_read_bytes = create_metric_per_engine(
            counter_read_bytes, per_engine_labelvalues
        )

        counter_write_bytes = self._counter_cls(
            name="vllm:estimated_write_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes written to memory per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_write_bytes = create_metric_per_engine(
            counter_write_bytes, per_engine_labelvalues
        )

    def observe(self, perf_stats: PerfStats, engine_idx: int = 0):
        if not (
            perf_stats.num_flops_per_gpu
            or perf_stats.num_read_bytes_per_gpu
            or perf_stats.num_write_bytes_per_gpu
        ):
            return
        self.counter_flops[engine_idx].inc(perf_stats.num_flops_per_gpu)
        self.counter_read_bytes[engine_idx].inc(perf_stats.num_read_bytes_per_gpu)
        self.counter_write_bytes[engine_idx].inc(perf_stats.num_write_bytes_per_gpu)


## util functions


def get_required(obj: object, attr: str):
    """Get an attr from an object, or throw a InvalidComponentError if it's not set."""
    if not hasattr(obj, attr):
        raise InvalidComponent(f"Missing required attr {attr} in config")
    return getattr(obj, attr)


def getattr_from_list(obj: object, attrs: list[str], default: object = None):
    """Try to get the first attr that exists in the object
    from a list of attrs. Otherwise return None."""
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return default
