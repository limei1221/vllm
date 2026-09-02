# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    get_args,
)

import numpy as np
import torch
from typing_extensions import runtime_checkable

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d

if TYPE_CHECKING:
    pass

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)

logger = init_logger(__name__)
KVCacheLayoutType = Literal["NHD", "HND"]
_KV_CACHE_LAYOUT_OVERRIDE: KVCacheLayoutType | None = None

PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0


def fill_mm_prefix_query_ranges(
    out: np.ndarray,
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> int:
    """Map each scheduled query token to the mm_prefix range containing it.

    Writes into ``out``, a caller-owned ``(max_num_batched_tokens, 2)`` int32
    staging buffer, and returns the number of rows written (0 if no range
    covers any scheduled query token, in which case ``out`` is untouched and
    the caller should skip the mask_mod entirely). Row ``i`` holds the absolute
    ``[start, end]`` bounds of the bidirectional range that query token ``i``
    belongs to, or ``(-1, -1)`` when it is outside every range.

    mm_prefix ranges never overlap, so "query and key share a range" is
    equivalent to "the key lies inside the query's own range". The kernel
    therefore needs no key-side lookup, and this metadata is sized by scheduled
    query tokens rather than by context length -- bounded by
    ``max_num_batched_tokens`` instead of ``num_seqs * max_seq_len``.

    Ranges are absolute prompt positions and may extend past the tokens
    scheduled so far under chunked prefill; the portion outside the current
    chunk is simply not recorded. Degenerate ranges (``start >= end``) are
    skipped to match the Triton path's ``start < end`` validity check.

    ``seq_lens_cpu`` only needs to be exact for prefill rows, since mm_prefix
    ranges cover prompt tokens: an over-estimate on a decode row shifts that
    row's query position further past every range, which still matches nothing.
    """
    if mm_prefix_range is None:
        return 0

    query_start_loc = query_start_loc_cpu.numpy()
    num_actual_tokens = int(query_start_loc[-1])
    if num_actual_tokens <= 0:
        return 0
    assert num_actual_tokens <= out.shape[0], (
        f"mm_prefix staging buffer holds {out.shape[0]} tokens, got {num_actual_tokens}"
    )

    # Resolve every span before touching `out`, so batches whose ranges all
    # fall outside the scheduled tokens skip the fill entirely.
    spans: list[tuple[int, int, int, int]] = []
    for req_idx, req_ranges in mm_prefix_range.items():
        if not req_ranges:
            continue
        token_start = int(query_start_loc[req_idx])
        query_len = int(query_start_loc[req_idx + 1]) - token_start
        if query_len <= 0:
            continue
        # Absolute position of this request's first scheduled query token.
        context_len = int(seq_lens_cpu[req_idx]) - query_len
        for start, end in req_ranges:
            if start >= end:
                continue
            first = max(start - context_len, 0)
            last = min(end - context_len, query_len - 1)
            if first > last:
                continue
            spans.append((token_start + first, token_start + last + 1, start, end))

    if not spans:
        return 0

    out[:num_actual_tokens] = -1
    for row_start, row_end, start, end in spans:
        out[row_start:row_end] = (start, end)
    return num_actual_tokens


def is_valid_kv_cache_layout(value: str) -> bool:
    return value in get_args(KVCacheLayoutType)


@functools.lru_cache
def get_kv_cache_layout():
    # Format specified by the code.
    global _KV_CACHE_LAYOUT_OVERRIDE

    cache_layout: Literal["NHD", "HND"] | None = None
    if _KV_CACHE_LAYOUT_OVERRIDE is not None:
        cache_layout = _KV_CACHE_LAYOUT_OVERRIDE
        logger.debug_once(
            "`_KV_CACHE_LAYOUT_OVERRIDE` variable detected. "
            "Setting KV cache layout to %s.",
            cache_layout,
        )
        return cache_layout

    # Format specified by the user.
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    # When neither the user nor the override specified a layout, get default
    if cache_layout is None:
        cache_layout = get_kv_connector_cache_layout()
    else:
        assert is_valid_kv_cache_layout(cache_layout)
        logger.info_once(
            "`VLLM_KV_CACHE_LAYOUT` environment variable "
            "detected. Setting KV cache layout to %s.",
            cache_layout,
        )
    return cache_layout


def set_kv_cache_layout(cache_layout: KVCacheLayoutType | None):
    global _KV_CACHE_LAYOUT_OVERRIDE
    _KV_CACHE_LAYOUT_OVERRIDE = cache_layout
    get_kv_cache_layout.cache_clear()


@dataclass
class PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters. Should not be used for
    trtllm-gen backend since it supports different values for the following
    hyperparameters.
    """

    window_left: int
    logits_soft_cap: float | None
    sm_scale: float
    has_sinks: bool = False
    # has same params for all layers
    has_same_window_lefts: bool | None = field(default=None, compare=False)
    has_same_all_params: bool | None = field(default=None, compare=False)


def get_num_attention_heads_from_layers(
    vllm_config: VllmConfig, layer_names: list[str]
) -> int | None:
    """Per-TP-rank ``num_heads`` shared by the named Attention layers.

    Use in metadata builders whose plan-time allocations depend on the
    head count: the model-wide ``get_num_attention_heads()`` is wrong
    for models with non-uniform per-layer head counts. All layers in
    one attention group must agree on ``num_heads``; this is asserted.
    Returns ``None`` when no matching Attention layer is found.
    """
    attn_layers = get_layers_from_vllm_config(
        vllm_config,
        AttentionLayerBase,  # type: ignore[type-abstract]
        layer_names,
    )
    if not attn_layers:
        return None
    heads = {layer.impl.num_heads for layer in attn_layers.values()}
    assert len(heads) == 1, (
        f"All layers in one attention group must share num_heads; "
        f"got {heads} for {layer_names}."
    )
    return heads.pop()


#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    common_attn_metadata: CommonAttentionMetadata,
    block_size: int = 0,
) -> tuple[CommonAttentionMetadata, Callable[[torch.Tensor], torch.Tensor]]:
    query_start_loc_np = common_attn_metadata.query_start_loc_cpu.numpy()
    seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
    block_table = common_attn_metadata.block_table_tensor
    device = common_attn_metadata.query_start_loc.device

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: make a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    num_computed_tokens_local = seqlens_k_local - seqlens_q_local

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // block_size
    assert attn_chunk_size % block_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not divisible by block_size {block_size}"
    )
    pages_per_local_batch = attn_chunk_size // block_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming block_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = block_starts[:, None] + np.arange(
        pages_per_local_batch, dtype=np.int32
    )
    block_indices = block_indices.reshape(-1).clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    # NOTE: https://github.com/pytorch/pytorch/pull/160256 causes performance
    # regression when using numpy arrays (batch and block indices) to index into
    # torch tensor (block_table). As a workaround, convert numpy arrays to torch
    # tensor first, which recovers perf.
    # Upload the index tensors to the block_table's device up-front so that the
    # fancy indexing below doesn't implicitly force a synchronous H2D copy.
    batch_indices_torch = async_tensor_h2d(batch_indices, device=device)
    block_indices_torch = async_tensor_h2d(block_indices, device=device)

    # Save as a lambda so we can return this for update_block_table
    make_block_table = lambda block_table: block_table[
        batch_indices_torch, block_indices_torch
    ].view(virtual_batches, -1)
    block_table_local = make_block_table(block_table)

    query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
    seq_lens_cpu = torch.from_numpy(seqlens_k_local)
    max_seq_len = int(seq_lens_cpu.max())

    return CommonAttentionMetadata(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=async_tensor_h2d(query_start_loc_cpu, device=device),
        seq_lens=async_tensor_h2d(seq_lens_cpu, device=device),
        num_reqs=len(seq_lens_cpu),
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=seqlens_q_local.max(),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_local,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local),
    ), make_block_table


def split_decodes_prefills_and_extends(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_extends: The number of extend requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_extend_tokens: The number of tokens in the extend requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, 0, num_tokens, 0, 0

    # Upper bound is exact for prefill rows; decode rows still satisfy
    # seq_len > query_len under the optimistic bound, so `seq_lens ==
    # query_lens` identifies prefills correctly either way.
    assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill_or_extend = query_lens > decode_threshold
    is_prefill = (seq_lens == query_lens) & is_prefill_or_extend
    first_extend = is_prefill_or_extend.int().argmax(dim=-1).item()
    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_extend
    num_decode_tokens = query_start_loc[first_extend].item()
    if not torch.any(is_prefill_or_extend):
        return (num_decodes, 0, 0, num_decode_tokens, 0, 0)

    num_prefills_or_extends = num_reqs - num_decodes
    num_prefill_or_extend_tokens = num_tokens - num_decode_tokens
    if not torch.any(is_prefill):
        return (
            num_decodes,
            num_prefills_or_extends,
            0,
            num_decode_tokens,
            num_prefill_or_extend_tokens,
            0,
        )

    num_extends = first_prefill - num_decodes
    num_prefills = num_reqs - first_prefill

    num_prefill_tokens = num_tokens - query_start_loc[first_prefill]
    num_extend_tokens = num_prefill_or_extend_tokens - num_prefill_tokens
    return (
        num_decodes,
        num_extends,
        num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        num_prefill_tokens,
    )


def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
    treat_short_extends_as_decodes: bool = True,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    The batch is expected to be ordered as:
        decode → short_extend → long_extend → prefill

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.
        require_uniform: If True, requires that all decode requests have the
            same query length. When set, some queries may be considered prefills
            even if they are <= decode_threshold, in order to ensure uniformity.
        treat_short_extends_as_decodes: If True (default), short extends
            (query_len <= threshold but still prefilling) are counted as
            decodes. If False, they are counted as prefills.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if query_lens[0].item() > decode_threshold:
        # first request is not decode, so no decode requests
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        # check if we are in a padded uniform batch; this is used for full-CGs, some
        # requests may have a query length of 0 but since they are padding its fine
        # to treat them as decodes (ensures num_decodes matches the captured size)
        if treat_short_extends_as_decodes and torch.all(
            (query_lens == query_lens[0]) | (query_lens == 0)
        ):
            return num_reqs, 0, num_tokens, 0  # all decodes
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not treat_short_extends_as_decodes:
        assert common_attn_metadata.is_prefilling is not None
        is_prefill |= common_attn_metadata.is_prefilling

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def reshape_query_for_spec_decode(query: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Reshapes the query tensor for the specified batch size, so that
    it has shape (batch_size, seq_len, num_heads, head_dim).
    """
    assert query.dim() == 3, f"query must be 3D, got {query.dim()}D"
    total_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    assert total_tokens % batch_size == 0, (
        f"{total_tokens=} is not divisible by {batch_size=}"
    )
    seq_len = total_tokens // batch_size
    return query.view(batch_size, seq_len, num_heads, head_dim)


def reshape_attn_output_for_spec_decode(attn_output: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the attention output tensor, so that
    the batch_size and seq_len dimensions are combined.
    """
    if attn_output.dim() == 3:
        # Already in the correct shape
        return attn_output
    assert attn_output.dim() == 4, f"attn_output must be 4D, got {attn_output.dim()}D"
    total_tokens = attn_output.shape[0] * attn_output.shape[1]
    return attn_output.view(total_tokens, attn_output.shape[2], attn_output.shape[3])


@runtime_checkable
class KVSharingFastPrefillMetadata(Protocol):
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None


def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """While using dcp, kv_cache size stored on each rank may be different,
    use this function to calculate split decode seq_lens of each dcp rank.
    Only consider dcp now, we can extend the case of cp based on this.
    """
    seq_lens_i32 = seq_lens.to(torch.int32)
    if dcp_rank is None:
        rank_offsets = torch.arange(
            dcp_size,
            dtype=torch.int32,
            device=seq_lens.device,
        ).view(
            *((1,) * seq_lens_i32.dim()),
            dcp_size,
        )
        seq_lens_tiled = seq_lens_i32.unsqueeze(-1)
    else:
        rank_offsets = torch.tensor(dcp_rank, dtype=torch.int32, device=seq_lens.device)
        seq_lens_tiled = seq_lens_i32
    base = (
        seq_lens_tiled
        // cp_kv_cache_interleave_size
        // dcp_size
        * cp_kv_cache_interleave_size
    )
    remainder = seq_lens_tiled - base * dcp_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    dcp_local_seq_lens = base + remainder
    return dcp_local_seq_lens
