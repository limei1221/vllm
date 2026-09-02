# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from packaging.version import Version
from transformers import PretrainedConfig
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.logger import init_logger

logger = init_logger(__name__)


def adapt_config_dict(
    config_dict: dict[str, Any],
    defaults: dict[str, Any],
) -> PretrainedConfig:
    config_dict = _remap_general_mistral_args(config_dict)
    config_dict = _remap_mistral_sliding_window(config_dict)

    if bool(config_dict.get("quantization")):
        config_dict = _remap_mistral_quantization_args(config_dict)

    is_mla = bool(config_dict.get("qk_nope_head_dim"))
    if is_mla:
        config_dict = _remap_mistral_mla_args(config_dict)

    is_moe = bool(config_dict.get("moe"))
    is_mistral_large_3 = (
        is_moe and (config_dict["moe"].get("num_shared_experts") or 0) > 0
    )
    if config_dict.get("model_type") == "mamba":
        config_dict["architectures"] = ["Mamba2ForCausalLM"]
    elif is_moe and is_mistral_large_3:
        config_dict = _remap_moe_args(config_dict)
        config_dict["model_type"] = "deepseek_v3"
        config_dict["architectures"] = ["MistralLarge3ForCausalLM"]

        assert "llama_4_scaling" in config_dict, (
            "MistralLarge3 expect llama4 scaling config."
        )
        llama_4_scaling_config_keys = ["original_max_position_embeddings", "beta"]
        assert all(
            [
                key in config_dict["llama_4_scaling"]
                for key in llama_4_scaling_config_keys
            ]
        ), (
            "llama_4_scaling config should define the keys: "
            f"{','.join(llama_4_scaling_config_keys)}"
        )
    elif is_moe:
        config_dict["architectures"] = ["MixtralForCausalLM"]
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]

    if bool(config_dict.get("yarn")):
        config_dict = _remap_mistral_yarn_args(config_dict)

    if bool(config_dict.get("llama_4_scaling")):
        llama_4_scaling_config_keys = ["original_max_position_embeddings", "beta"]
        assert all(
            [
                key in config_dict["llama_4_scaling"]
                for key in llama_4_scaling_config_keys
            ]
        ), (
            "llama_4_scaling config should define the keys: "
            f"{','.join(llama_4_scaling_config_keys)}"
        )

    for k, v in defaults.items():
        config_dict.setdefault(k, v)

    config = PretrainedConfig.from_dict(config_dict)

    logger.debug("Initialized config %s", config)

    return config


def _remap_mistral_yarn_args(config: dict) -> dict:
    yarn_config_map = {
        "factor": ("factor", float),
        "original_max_position_embeddings": ("original_max_position_embeddings", int),
        "beta": ("beta_fast", float),
        "alpha": ("beta_slow", float),
        "apply_scale": ("apply_yarn_scaling", bool),
    }

    yarn_config = config.get("yarn") or {}
    config["rope_parameters"] = {
        "rope_type": "yarn",
        "mscale_all_dim": 1,
    }

    if rope_theta := config.pop("rope_theta", None):
        config["rope_parameters"]["rope_theta"] = rope_theta

    for old_name, (new_name, cast) in yarn_config_map.items():
        if old_name in yarn_config:
            # Cast to remove Transformers > v5 type warnings
            config["rope_parameters"][new_name] = cast(yarn_config.pop(old_name))

    # Ignore apply_yarn_scaling in Transformers > v5 RoPE validation to remove warnings
    if Version(TRANSFORMERS_VERSION) >= Version("5.3.0.dev0"):
        config["ignore_keys_at_rope_validation"] = {"apply_yarn_scaling"}

    assert len(yarn_config) == 0, f"Unparsed yarn config: {yarn_config}"

    return config


def _remap_general_mistral_args(config: dict) -> dict:
    # Mistral key -> HF key
    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "hidden_dim": "intermediate_size",
    }
    # HF key -> (Mistral key, default value)
    top_level_mapping_with_default = {
        "model_type": ("model_type", "transformer"),
        "hidden_act": ("activation", "silu"),
        "tie_word_embeddings": ("tied_embeddings", False),
        "max_seq_len": ("max_seq_len", config.get("max_position_embeddings", 128_000)),
        "max_position_embeddings": ("max_position_embeddings", 128_000),
        "dtype": ("dtype", config.get("dtype")),
    }

    for key, new_key in config_mapping.items():
        if key in config:
            config[new_key] = config.pop(key)

    for new_key, (key, default_value) in top_level_mapping_with_default.items():
        config[new_key] = config.pop(key, default_value)

    return config


def _remap_mistral_sliding_window(config: dict) -> dict:
    # Remap sliding_window (list) -> layer_types (list) + sliding window (int)
    # for HF compatibility
    # Mistral configs may define sliding_window as list[int]. Convert it
    # to int and add the layer_types list[str] to make it HF compatible
    if sliding_window := config.get("sliding_window"):
        if isinstance(sliding_window, list):
            pattern_repeats = config["num_hidden_layers"] // len(sliding_window)
            layer_types = sliding_window * pattern_repeats
            config["layer_types"] = [
                "full_attention" if layer_type is None else "sliding_attention"
                for layer_type in layer_types
            ]
            assert len(set(sliding_window) - {None}) <= 1, sliding_window
            config["sliding_window"] = next(filter(None, sliding_window), None)
        elif isinstance(sliding_window, int) and config.get("layer_types") is None:
            config["layer_types"] = ["sliding_attention"] * config["num_hidden_layers"]
        else:
            raise ValueError(f"Unsupported sliding_window type: {sliding_window}")

    return config


def _remap_mistral_quantization_args(config: dict) -> dict:
    if config.get("quantization"):
        quantization = config.pop("quantization", {})
        if quantization.get("qformat_weight") == "fp8_e4m3":
            qscheme_act = quantization.get("qscheme_act")
            assert qscheme_act in ("NO_SCALES", "TENSOR", None), (
                "Only NO_SCALES and TENSOR (default) are supported for qscheme_act"
            )
            is_dynamic = qscheme_act == "NO_SCALES"
            config["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic" if is_dynamic else "static",
            }
        elif (
            str(quantization.get("quant_method", "")).lower().replace("_", "-")
            == "compressed-tensors"
        ):
            # Pass through compressed-tensors config, while normalizing
            # quant_method to the canonical community spelling.
            quantization["quant_method"] = "compressed-tensors"
            config["quantization_config"] = quantization
        else:
            raise ValueError(f"Found unknown quantization='{quantization}' in config")

    return config


def _remap_moe_args(config: dict) -> dict:
    moe_config_map = {
        "route_every_n": "moe_layer_freq",
        "first_k_dense_replace": "first_k_dense_replace",
        "num_experts_per_tok": "num_experts_per_tok",
        "num_experts": "n_routed_experts",
        "expert_hidden_dim": "moe_intermediate_size",
        "routed_scale": "routed_scaling_factor",
        "num_shared_experts": "n_shared_experts",
        "num_expert_groups": "n_group",
        "num_expert_groups_per_tok": "topk_group",
    }
    moe_config = config.get("moe", {})
    for old_name, new_name in moe_config_map.items():
        if old_name in moe_config:
            value = moe_config.pop(old_name)
            config[new_name] = value

    config["topk_method"] = None
    config["norm_topk_prob"] = True
    config["scoring_func"] = "softmax"

    return config


def _remap_mistral_mla_args(config: dict) -> dict:
    if not config.get("moe"):
        moe = {
            "num_experts": 1,
            "first_k_dense_replace": config.get("num_hidden_layers"),
            "route_every_n": 1,
            "num_shared_experts": 1,
            "expert_hidden_dim": config.get("intermediate_size"),
            "num_experts_per_tok": 1,
            "routed_scale": 1.0,
            "renorm_strategy": "WEIGHTS",
            "use_load_balancing_bias": False,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
        }
        config["moe"] = moe
    return config
