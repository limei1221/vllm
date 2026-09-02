<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
A lean vLLM: DeepSeek on Hopper, and nothing else
</h3>

---

## About

This is a **stripped-down fork of [vLLM](https://github.com/vllm-project/vllm)**, cut
down to a single coherent inference path so the engine can be read end to end. It is a
learning and research tree, **not** a drop-in replacement for upstream vLLM — most of
what upstream supports has been deliberately deleted.

If you want vLLM for production, use [upstream](https://github.com/vllm-project/vllm).

## What this tree supports

- **Models**: DeepSeek V2/V3 only, plus MTP and EAGLE speculative decoding
- **Hardware**: NVIDIA Hopper (SM90) only, CUDA only
- **Quantization**: BF16 and FP8 only (DeepGEMM for FP8 MoE, Triton for BF16)
- **Attention**: FlashAttention for prefill, FlashMLA for decode
- **Weights**: safetensors only
- **Parallelism**: local TP, PP, DP, EP, PCP, DCP — single node
- **Executors**: `MultiprocExecutor` (multi-GPU), `UniProcExecutor` (single-GPU)
- **All2all**: `allgather_reducescatter` only
- **Server**: four HTTP routes — `/v1/chat/completions`, `/v1/completions`,
  `/v1/models`, `/health`
- **Offline API**: `LLM.generate()` and `LLM.chat()`

## What has been removed

LoRA · multimodal · pooling and embedding models · structured outputs and grammar
backends · beam search · the model zoo beyond DeepSeek · non-FP8 quantization
(GPTQ/AWQ/GGUF/INT8/…) · ROCm, TPU, CPU and every non-CUDA backend · Ray and other
alternate executors · KV-connector backends and disaggregated prefill · NIXL and the
other all2all backends · the Rust frontend · the benchmark suite.

Attempting to use a removed feature should raise a clear error rather than silently
misbehave. If you find one that fails quietly, that is a bug worth reporting.

## Getting started

Build from source — there is no published wheel for this fork:

```bash
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Serve a model:

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --tensor-parallel-size 2
```

See [`examples/lean/`](examples/lean/) for a runnable offline script and a serving
script, and [`docs/lean/architecture.md`](docs/lean/architecture.md) for the request
flow and the design decisions behind the cuts.

## Keeping the tree lean

Two pre-commit hooks guard the invariants:

- `lean-tree-guard` — [`tools/check_lean_tree.py`](tools/check_lean_tree.py) fails if a
  removed subsystem reappears by path or by name, or if the tree grows past its
  line budget.
- `mypy-lean-tree` — type-checks the whole `vllm` package, not just changed files, so a
  caller and callee cannot drift apart across separate commits.

```bash
uv pip install -r requirements/lint.txt && pre-commit install
pre-commit run --all-files
```

The guard also enforces a runtime line-count budget of 195,000 lines. The tree
currently sits under it, so `--budgets` runs as part of the hook and the number
acts as a ratchet rather than an aspiration.

## Upstream

vLLM was originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu)
at UC Berkeley and is maintained by a large open-source community. All credit for the
engine belongs there; this fork only deletes.

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
