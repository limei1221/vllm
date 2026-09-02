#pragma once

#include <optional>
#include <string>
#include <torch/library.h>
#include <tuple>

#include "core/scalar_type.hpp"

#include <vector>

// rms_norm and fused_add_rms_norm declarations also exist in
// csrc/libtorch_stable/ops.h (torch::stable ABI for CUDA). They remain here
// because the CPU build still uses these torch::Tensor declarations.
void rms_norm(torch::Tensor& out, torch::Tensor& input,
              std::optional<torch::Tensor> weight, double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        std::optional<torch::Tensor> weight, double epsilon);

// rotary_embedding also exist in csrc/libtorch_stable/ops.h (torch::stable
// ABI for CUDA). It remains here because the CPU build still uses these
// torch::Tensor declarations.
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void silu_and_mul_clamp(torch::Tensor& out, torch::Tensor& input, double limit,
                        double alpha = 1.0, double beta = 0.0);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void relu_squared(torch::Tensor& out, torch::Tensor& input);

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

using fptr_t = int64_t;
