# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


logger = init_logger(__name__)


class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        return

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        return


class DeepseekV32ForCausalLM(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        hf_config = vllm_config.model_config.hf_config

        # Mirror the check in vllm/model_executor/models/deepseek_v2.py
        is_v32 = hasattr(hf_config, "index_topk")
        assert is_v32

        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "bfloat16":
            cache_config.cache_dtype = "auto"
            logger.info("Using bfloat16 kv-cache for DeepSeekV3.2")


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {
    "DeepseekV32ForCausalLM": DeepseekV32ForCausalLM,
}
