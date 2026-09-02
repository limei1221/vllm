# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.distributed import (
    get_ep_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_naive_dp_ep,
    make_moe_prepare_and_finalize_no_dp_ep,
)

logger = init_logger(__name__)


def get_ep_all2all_manager(eep_stage: bool = False) -> Any:
    if eep_stage:
        from vllm.distributed.elastic_ep.standby_state import get_standby_ep_group

        ep_group = get_standby_ep_group()
        assert ep_group is not None
        device_communicator = ep_group.device_communicator
    else:
        device_communicator = get_ep_group().device_communicator

    assert device_communicator is not None
    all2all_manager = device_communicator.all2all_manager
    assert all2all_manager is not None
    return all2all_manager


def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        moe_parallel_config: Fused MoE parallelization strategy configuration.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs
        and all2all backend.
        Original hidden size otherwise.
    """
    # allgather_reducescatter needs no hidden-size rounding.
    del act_dtype, moe_parallel_config
    return hidden_size


def maybe_make_prepare_finalize(
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    allow_new_interface: bool = False,
    use_monolithic: bool = False,
    eep_stage: bool = False,
) -> FusedMoEPrepareAndFinalize | None:
    if not moe.moe_parallel_config.use_all2all_kernels:
        if not allow_new_interface:
            return None

        # Opt-in XPU batched path: reorganize tokens into E x T x K locally
        # (no all-to-all) so BatchedTritonExperts (moe_mmk TD) can run.
        # For DP/TP case, fall back to naive P/F.
        if moe.moe_parallel_config.dp_size > 1:
            logger.info_once(
                "Detected DP deployment with no --enable-expert-parallel. "
                "Falling back to AllGather+ReduceScatter dispatch/combine."
            )
            all2all_manager = get_ep_all2all_manager(eep_stage)
            return make_moe_prepare_and_finalize_naive_dp_ep(
                is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
                num_dispatchers=all2all_manager.world_size,
                use_monolithic=use_monolithic,
            )
        else:
            return make_moe_prepare_and_finalize_no_dp_ep(use_monolithic)

    all2all_manager = get_ep_all2all_manager(eep_stage)

    prepare_finalize: FusedMoEPrepareAndFinalize | None = None

    if moe.use_ag_rs_all2all_kernels and allow_new_interface:
        prepare_finalize = make_moe_prepare_and_finalize_naive_dp_ep(
            use_monolithic=use_monolithic,
            is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
            num_dispatchers=all2all_manager.world_size,
        )

    return prepare_finalize
