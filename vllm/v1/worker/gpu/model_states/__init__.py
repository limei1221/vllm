# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    device: torch.device,
):
    # Let the model provide its own ModelState if it defines one.
    if hasattr(model, "get_model_state_cls"):
        cls = model.get_model_state_cls()
        return cls(vllm_config, model, device)

    # Encoder-only attention is non-causal and needs no KV cache.
    if any(
        layer.attn_type == AttentionType.ENCODER_ONLY
        for layer in get_layers_from_vllm_config(vllm_config, Attention).values()
    ):
        from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState

        return EncoderOnlyModelState(vllm_config, model, device)

    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState(vllm_config, model, device)
