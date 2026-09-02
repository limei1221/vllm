# Lean vLLM Architecture

This focused build of vLLM supports **DeepSeek V2/V3** inference on
**NVIDIA Hopper SM90** only, with local TP, PP, DP, EP, PCP, DCP, MTP,
and EAGLE.

## Request Flow

```text
Client → FastAPI (4 routes) → AsyncLLM → EngineCore → Scheduler → GPUModelRunner
                                                                          ↓
                                                    MultiprocExecutor → Workers
```

## Key Design Decisions

- **One executor**: `MultiprocExecutor` for multi-GPU, `UniProcExecutor` for single-GPU
- **One loader**: `SafetensorsModelLoader` for all weight loading
- **One model family**: DeepSeek V2/V3 with MTP and EAGLE
- **Two attention backends**: FlashAttention (prefill) + FlashMLA (decode)
- **Two MoE dtypes**: BF16 (Triton) + FP8 (DeepGEMM)
- **One all2all backend**: allgather_reducescatter
- **Four routes**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`
- **Two offline methods**: `LLM.generate()` and `LLM.chat()`

## Test Coverage Map

| Acceptance Criterion | Test File |
| --- | --- |
| Public imports | `tests/lean/test_public_contract.py` |
| DeepSeek architectures only | `tests/lean/test_model_resolution.py` |
| Hopper enforcement | `tests/lean/test_runtime_validation.py` |
| Safetensors-only loading | `tests/model_executor/model_loader/test_safetensors_only_loader.py` |
| Four routes only | `tests/lean/test_server_routes.py` |
| Local DP round-robin | `tests/entrypoints/openai/test_dp_supervisor.py` |
| Offline API surface | `tests/lean/test_offline_contract.py` |
| Speculative methods | `tests/lean/test_focused_cli.py` |
| Parallel topology | `tests/lean/test_parallel_topology.py` |
| Static EP | `tests/lean/test_static_expert_parallel.py` |
| Multi-GPU topologies | `tests/lean/test_multigpu_topologies.py` |
