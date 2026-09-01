# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm/AITER shim.

This build targets CUDA only, so the AITER kernels this module used to wrap
have been removed. The object is kept because ~25 device-agnostic modules
import it at module scope and gate their AITER paths on these predicates,
which are all False here. Nothing beyond a predicate is reachable on CUDA.
"""


class _RocmAiterOpsUnavailable:
    """Every AITER capability reports unavailable."""

    @staticmethod
    def are_gdn_triton_kernels_available() -> bool:
        return False

    @staticmethod
    def is_asm_fp4_gemm_dynamic_quant_enabled() -> bool:
        return False

    @staticmethod
    def is_custom_all_reduce_enabled() -> bool:
        return False

    @staticmethod
    def is_enabled() -> bool:
        return False

    @staticmethod
    def is_fp4bmm_enabled() -> bool:
        return False

    @staticmethod
    def is_fp8bmm_enabled() -> bool:
        return False

    @staticmethod
    def is_fused_moe_enabled() -> bool:
        return False

    @staticmethod
    def is_fused_moe_situv2_a8w4_enabled() -> bool:
        return False

    @staticmethod
    def is_fusion_moe_shared_experts_enabled() -> bool:
        return False

    @staticmethod
    def is_linear_fp8_enabled() -> bool:
        return False

    @staticmethod
    def is_mha_enabled() -> bool:
        return False

    @staticmethod
    def is_rdna_aiter_enabled() -> bool:
        return False

    @staticmethod
    def is_rdna_gdn_triton_kernels_available() -> bool:
        return False

    @staticmethod
    def is_tgemm_enabled() -> bool:
        return False

    @staticmethod
    def is_triton_gemm_afp4wfp4_presh_ws_tuned() -> bool:
        return False

    @staticmethod
    def is_triton_gemm_enabled() -> bool:
        return False

    @staticmethod
    def is_triton_rotary_embed_enabled() -> bool:
        return False

    @staticmethod
    def get_aiter_allreduce(*args, **kwargs):
        return None

    @staticmethod
    def get_fused_allreduce_rmsnorm_op(*args, **kwargs):
        return None

    @staticmethod
    def get_fused_allreduce_rmsnorm_quant_per_group_op(*args, **kwargs):
        return None

    @staticmethod
    def get_fused_allreduce_rmsnorm_quant_per_group_with_bf16_norm_op(*args, **kwargs):
        return None

    @staticmethod
    def get_group_quant_op(*args, **kwargs):
        return None

    @staticmethod
    def get_per_token_quant_op(*args, **kwargs):
        return None

    @staticmethod
    def get_rmsnorm_group_fused_quant_op(*args, **kwargs):
        return None

    @staticmethod
    def get_triton_add_rmsnorm_pad_op(*args, **kwargs):
        return None

    @staticmethod
    def get_triton_rotary_embedding_op(*args, **kwargs):
        return None

    @staticmethod
    def refresh_env_variables() -> None:
        return None

    @staticmethod
    def batched_gemm_a16wfp4(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'batched_gemm_a16wfp4' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def biased_grouped_topk(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'biased_grouped_topk' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'flash_attn_varlen_func' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def fp8_attn_wrapper(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'fp8_attn_wrapper' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def group_fp8_quant(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'group_fp8_quant' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def mla_decode_fwd(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'mla_decode_fwd' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def per_tensor_quant(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'per_tensor_quant' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def per_token_quant(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'per_token_quant' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def py(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'py' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def shuffle_mxfp8_moe_weights(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'shuffle_mxfp8_moe_weights' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def shuffle_scale_a16w4(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'shuffle_scale_a16w4' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def shuffle_weight(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'shuffle_weight' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def shuffle_weight_a16w4(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'shuffle_weight_a16w4' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def shuffle_weights(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'shuffle_weights' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def topk_sigmoid(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'topk_sigmoid' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def topk_softmax(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'topk_softmax' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def triton_fp8_bmm(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'triton_fp8_bmm' is ROCm-only and this build is CUDA-only."
        )

    @staticmethod
    def triton_rope_and_cache(*args, **kwargs):
        raise NotImplementedError(
            "AITER op 'triton_rope_and_cache' is ROCm-only and this build is CUDA-only."
        )


rocm_aiter_ops = _RocmAiterOpsUnavailable()


def is_aiter_found() -> bool:
    return False


def is_aiter_found_and_supported() -> bool:
    return False


def is_aiter_found_and_supported_on_rdna4() -> bool:
    return False


IS_AITER_FOUND = False
