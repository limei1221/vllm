# DeepSeek Serving Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `lean-vllm` to a runnable, optimized, Hopper-only DeepSeek V2/V3 serving and offline-inference vertical slice while retaining local TP, PP, DP, EP, PCP, DCP, MTP, and EAGLE.

**Architecture:** Online and offline entrypoints converge on the V1 scheduler, one local multiprocessing executor, and one Hopper DeepSeek model runner. Generic registries and interchangeable backends are replaced by explicit DeepSeek, safetensors, SM90, NCCL/Gloo, FlashAttention/FlashMLA, and DeepGEMM wiring; every unsupported path is deleted or rejected explicitly.

**Tech Stack:** Python 3.12, PyTorch 2.13, CUDA SM90, NCCL/Gloo, FastAPI, Hugging Face Transformers and Hub, safetensors, FlashAttention, FlashMLA, DeepGEMM, Triton, pytest, Ruff, pre-commit, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-01-deepseek-serving-vertical-slice-design.md`

## Global Constraints

- Run every Python command through `uv` and `.venv/bin/python`; never use system `python3`, bare `pip`, or bare `pip install`.
- Target Linux, one host, and NVIDIA Hopper SM90 only.
- Support DeepSeek V2/V3, DeepSeek MTP, EAGLE, and EAGLE3 only.
- Support Hugging Face safetensors from a local snapshot directory or Hub repository ID; support BF16 and pre-quantized FP8 only.
- Retain only `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, and `/metrics` online routes.
- Retain only `LLM`, `LLM.generate()`, and `LLM.chat()` offline APIs.
- Retain local TP, PP, DP, EP, PCP, and DCP with one local implementation each; remove Ray, external launchers, multi-node operation, elastic modes, and alternate transports.
- Retain continuous batching, paged KV cache, prefix caching, compilation, CUDA graphs, warmup, MTP, and EAGLE.
- Use Google-style docstrings and 88-character Python lines.
- Add behavior tests before implementation changes; extend nearby suites and fixtures before creating a new suite.
- Each commit must leave the retained path importable and include `Co-authored-by:` and `Signed-off-by:` trailers.
- Do not propose an upstream PR for this fork-wide reduction. If the user later requests a PR, run the mandatory duplicate-work checks from `AGENTS.md` first.

## Target File Map

| Responsibility | Retained target files |
| --- | --- |
| Public configuration | `vllm/config/{vllm,model,parallel,load,cache,scheduler,compilation,speculative}.py`, `vllm/engine/arg_utils.py` |
| Hopper platform contract | `vllm/platforms/{__init__,interface,cuda}.py` |
| Model resolution | `vllm/model_executor/models/{registry,deepseek_v2,deepseek_mtp,deepseek_eagle,deepseek_eagle3}.py` |
| Safetensors loading | `vllm/model_executor/model_loader/{__init__,base_loader,safetensors_loader,weight_utils,ep_weight_filter,mtp_validation}.py` |
| Local execution | `vllm/v1/executor/{abstract,uniproc_executor,multiproc_executor}.py` |
| Parallel groups | `vllm/distributed/{parallel_state,communication_op}.py`, `vllm/distributed/device_communicators/{base_device_communicator,cuda_communicator,shm_broadcast,all2all}.py` |
| Engine and scheduling | `vllm/v1/engine/`, `vllm/v1/core/`, `vllm/v1/kv_cache_interface.py` |
| Hopper model execution | `vllm/v1/worker/`, `vllm/v1/attention/`, `vllm/model_executor/layers/`, `vllm/model_executor/kernels/linear/` |
| Speculative decoding | `vllm/v1/spec_decode/{eagle,llm_base_proposer,metadata,metrics,utils,vocab_mapping}.py`, retained GPU speculator files |
| Online entrypoint | `vllm/entrypoints/cli/serve.py`, `vllm/entrypoints/openai/{api_server,cli_args,dp_supervisor}.py`, chat/completion/models routers |
| Offline entrypoint | `vllm/entrypoints/{llm,offline_utils,chat_utils}.py` |
| Focused static guard | `tools/check_lean_tree.py`, `tests/lean/test_tree_contract.py` |
| Focused documentation | `docs/lean/architecture.md`, `docs/lean/reading-order.md`, `docs/lean/parallelism.md`, `docs/lean/hopper-kernels.md` |

---

## Phase 1: Restore a Runnable Local Multi-GPU Baseline

### Task 1: Add executable contract characterization

**Files:**

- Create: `tests/lean/__init__.py`
- Create: `tests/lean/test_public_contract.py`

**Interfaces:**

- Consumes: current public imports and model registry.
- Produces: characterization tests for the retained public roots and six supported architecture names.

- [ ] **Step 1: Write the failing public-import test**

```python
def test_retained_public_roots_import() -> None:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.api_server import build_app
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    assert LLM is not None
    assert SamplingParams is not None
    assert build_app is not None
    assert MultiprocExecutor is not None
```

- [ ] **Step 2: Add the exact architecture-contract test**

```python
def test_only_deepseek_architectures_are_registered() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    assert set(ModelRegistry.get_supported_archs()) == {
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepSeekMTPModel",
        "EagleDeepSeekMTPModel",
        "Eagle3DeepseekV2ForCausalLM",
        "Eagle3DeepseekV3ForCausalLM",
    }
```

- [ ] **Step 3: Run the tests and capture the expected baseline failure**

Run: `.venv/bin/python -m pytest tests/lean/test_public_contract.py -v`

Expected: import collection fails because `vllm.v1.executor.multiproc_executor` is missing. If `.venv` is absent, first run `uv venv --python 3.12`, `uv pip install -r requirements/lint.txt`, `uv pip install -r requirements/test/cuda.in`, `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`, and `pre-commit install`.

- [ ] **Step 4: Commit the characterization test with the executor restoration in Task 2**

Do not commit a deliberately broken tree; stage these tests together with Task 2.

### Task 2: Restore and immediately localize multiprocessing execution

**Files:**

- Create: `vllm/v1/executor/multiproc_executor.py`
- Modify: `vllm/v1/executor/__init__.py`
- Modify: `vllm/v1/executor/abstract.py`
- Modify: `tests/v1/executor/test_executor.py`
- Modify: `tests/v1/executor/test_multiproc_executor_timeout.py`
- Test: `tests/lean/test_public_contract.py`

**Interfaces:**

- Consumes: `Executor`, `WorkerWrapperBase`, shared-memory `MessageQueue`, and local `ParallelConfig` ranks.
- Produces: `MultiprocExecutor(Executor)`, `FutureWrapper`, and locally spawned `WorkerProc` ranks; `Executor.get_class()` returns `UniProcExecutor` for one model rank and `MultiprocExecutor` otherwise.

- [ ] **Step 1: Add focused executor-selection tests**

```python
from types import SimpleNamespace


def test_executor_class_uses_uni_for_one_rank() -> None:
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=1))
    assert Executor.get_class(config) is UniProcExecutor


def test_executor_class_uses_local_mp() -> None:
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))
    assert Executor.get_class(config) is MultiprocExecutor
```

- [ ] **Step 2: Run executor tests to verify the missing local backend**

Run: `.venv/bin/python -m pytest tests/lean/test_public_contract.py tests/v1/executor/test_executor.py -v`

Expected: FAIL during import of `multiproc_executor`.

- [ ] **Step 3: Restore the behavioral reference and strip unsupported branches in the same patch**

Use `git show main:vllm/v1/executor/multiproc_executor.py` as a read-only reference. Retain `FutureWrapper`, `MultiprocExecutor`, `WorkerProc`, shared-memory request/response queues, worker health monitoring, coordinated shutdown, async scheduling, direct local-rank CUDA assignment, and TP/PP/PCP/DCP/EP group result aggregation. Remove Ray integration, remote-node message queues, node-rank branches, network-device routing, NIXL hooks, tracing setup, NUMA binding, custom executor qualnames, EC/KV aggregators, LoRA RPC helpers, sleep/wake RPC, and distributed reconfiguration.

The class selector must be direct:

```python
if parallel_config.world_size == 1:
    return UniProcExecutor
return MultiprocExecutor
```

- [ ] **Step 4: Export only the two local executors**

```python
from .abstract import Executor
from .multiproc_executor import MultiprocExecutor
from .uniproc_executor import UniProcExecutor

__all__ = ["Executor", "MultiprocExecutor", "UniProcExecutor"]
```

- [ ] **Step 5: Run focused executor tests**

Run: `.venv/bin/python -m pytest tests/lean/test_public_contract.py tests/v1/executor/test_executor.py tests/v1/executor/test_multiproc_executor_timeout.py -v`

Expected: PASS; no test references `ExecutorWithExternalLauncher`, custom executor classes, Ray, EC transfer, or KV transfer.

- [ ] **Step 6: Commit the runnable baseline**

```bash
git add tests/lean tests/models/registry.py tests/v1/executor vllm/v1/executor
git commit -m "Restore local multiprocess execution"
```

### Task 3: Enforce Hopper and local rank topology before worker startup

**Files:**

- Modify: `vllm/platforms/cuda.py`
- Modify: `vllm/config/parallel.py`
- Modify: `vllm/config/vllm.py`
- Modify: `vllm/engine/arg_utils.py`
- Create: `tests/lean/test_runtime_validation.py`
- Modify: `tests/cuda/test_cuda_context.py`

**Interfaces:**

- Consumes: `DeviceCapability`, visible CUDA device count, and parallel sizes.
- Produces: `CudaPlatform.verify_hopper() -> None` and `ParallelConfig.local_gpu_count -> int`.

- [ ] **Step 1: Write hardware and topology failures**

```python
def test_rejects_non_hopper(monkeypatch) -> None:
    monkeypatch.setattr(CudaPlatform, "get_device_capability", lambda *_: DeviceCapability(8, 0))
    with pytest.raises(RuntimeError, match="SM90 Hopper"):
        CudaPlatform.verify_hopper()


def test_local_gpu_count_multiplies_model_and_data_ranks() -> None:
    config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        prefill_context_parallel_size=2,
        data_parallel_size=2,
    )
    assert config.local_gpu_count == 16
```

- [ ] **Step 2: Run validation tests and verify failure**

Run: `.venv/bin/python -m pytest tests/lean/test_runtime_validation.py -v`

Expected: FAIL because `verify_hopper` and `local_gpu_count` do not exist.

- [ ] **Step 3: Implement the explicit validation contract**

```python
@classmethod
def verify_hopper(cls) -> None:
    capability = cls.get_device_capability()
    if capability != DeviceCapability(9, 0):
        raise RuntimeError(
            f"This focused vLLM build requires NVIDIA Hopper SM90; got {capability}."
        )
```

Define `local_gpu_count` as `TP * PP * PCP * DP`; require DCP to divide TP; require all sizes to be positive; reject `nnodes != 1`, Ray, external launcher, elastic EP, EPLB, remote DP, and alternate all-to-all names during `ParallelConfig.__post_init__`.

- [ ] **Step 4: Validate before engine subprocess creation**

Call `current_platform.verify_hopper()` and compare `current_platform.device_count()` with `parallel_config.local_gpu_count` in `EngineArgs.create_engine_config()` before returning `VllmConfig`.

- [ ] **Step 5: Run validation and configuration suites**

Run: `.venv/bin/python -m pytest tests/lean/test_runtime_validation.py tests/cuda/test_cuda_context.py tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/platforms/cuda.py vllm/config/parallel.py vllm/config/vllm.py vllm/engine/arg_utils.py tests/lean/test_runtime_validation.py tests/cuda/test_cuda_context.py tests/test_config.py
git commit -m "Enforce local Hopper runtime topology"
```

## Phase 2: Make Models, Loading, and Configuration Explicit

### Task 4: Replace the loader registry with one safetensors loader

**Files:**

- Create: `vllm/model_executor/model_loader/safetensors_loader.py`
- Modify: `vllm/model_executor/model_loader/__init__.py`
- Modify: `vllm/model_executor/model_loader/base_loader.py`
- Modify: `vllm/model_executor/model_loader/weight_utils.py`
- Modify: `vllm/config/load.py`
- Create: `tests/model_executor/model_loader/test_safetensors_only_loader.py`
- Delete: unsupported loader files and their test directories under `vllm/model_executor/model_loader/` and `tests/model_executor/model_loader/`.

**Interfaces:**

- Consumes: `LoadConfig`, Hugging Face Hub snapshot download, safetensors index, rank-local expert filtering.
- Produces: `ResolvedSafetensors(directory: Path, files: tuple[Path, ...])`, `SafetensorsModelLoader.load_model(...) -> nn.Module`, and `get_model(...)` with no loader registry.

- [ ] **Step 1: Test local, Hub, and rejected formats**

```python
@pytest.fixture
def safetensors_snapshot(tmp_path: Path) -> Path:
    (tmp_path / "model.safetensors").touch()
    return tmp_path


def test_load_config_accepts_only_safetensors() -> None:
    assert LoadConfig().load_format == "safetensors"
    with pytest.raises(ValueError, match="only safetensors"):
        LoadConfig(load_format="pt")


def test_missing_safetensors_is_explicit(tmp_path) -> None:
    loader = SafetensorsModelLoader(LoadConfig())
    with pytest.raises(RuntimeError, match="No safetensors weights"):
        loader.prepare_weights(str(tmp_path), revision=None)


def test_hub_id_resolves_snapshot(monkeypatch, safetensors_snapshot) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.safetensors_loader.snapshot_download",
        lambda **_: str(safetensors_snapshot),
    )
    loader = SafetensorsModelLoader(LoadConfig())
    resolved = loader.prepare_weights("org/deepseek-test", revision="main")
    assert resolved.directory == safetensors_snapshot
    assert all(path.suffix == ".safetensors" for path in resolved.files)
```

- [ ] **Step 2: Verify failures**

Run: `.venv/bin/python -m pytest tests/model_executor/model_loader/test_safetensors_only_loader.py -v`

Expected: FAIL because the focused loader does not exist and `LoadConfig` defaults to `auto`.

- [ ] **Step 3: Implement a safetensors-only source**

```python
@dataclass(frozen=True)
class SafetensorsSource:
    model_or_path: str
    revision: str | None
    prefix: str = ""


@dataclass(frozen=True)
class ResolvedSafetensors:
    directory: Path
    files: tuple[Path, ...]
```

Move only safetensors discovery, Hub download, index filtering, lazy/eager/prefetch iteration, MTP validation, and EP weight filtering from `DefaultModelLoader`. Delete `.bin`, `.pt`, npcache, Mistral-consolidated, ModelScope, torchao reconstruction, custom loader registration, reload, tensorizer, RunAI, ModelExpress, dummy, and sharded-state paths.

- [ ] **Step 4: Wire direct construction**

```python
def get_model(*, vllm_config: VllmConfig, model_config=None, prefix="") -> nn.Module:
    return SafetensorsModelLoader(vllm_config.load_config).load_model(
        vllm_config=vllm_config,
        model_config=model_config or vllm_config.model_config,
        prefix=prefix,
    )
```

- [ ] **Step 5: Run focused loader tests**

Run: `.venv/bin/python -m pytest tests/model_executor/model_loader/test_safetensors_only_loader.py tests/model_executor/model_loader/test_filter_duplicate_safetensors.py tests/model_executor/model_loader/test_mtp_validation.py tests/model_executor/model_loader/test_ep_weight_filter.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/config/load.py vllm/model_executor/model_loader tests/model_executor/model_loader
git commit -m "Reduce model loading to safetensors"
```

### Task 5: Collapse model resolution to DeepSeek classes

**Files:**

- Modify: `vllm/model_executor/models/registry.py`
- Modify: `vllm/model_executor/model_loader/utils.py`
- Modify: `vllm/transformers_utils/config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/models/registry.py`
- Create: `tests/lean/test_model_resolution.py`

**Interfaces:**

- Consumes: Hugging Face `architectures` strings.
- Produces: `resolve_deepseek_model_class(architecture: str) -> type[nn.Module]` and the existing `ModelRegistry` facade limited to six explicit lazy entries.

- [ ] **Step 1: Write direct-resolution behavior**

```python
@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_resolves_supported_architecture(architecture: str) -> None:
    assert resolve_deepseek_model_class(architecture).__name__


def test_rejects_other_architecture() -> None:
    with pytest.raises(ValueError, match="DeepSeek V2/V3"):
        resolve_deepseek_model_class("LlamaForCausalLM")
```

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/python -m pytest tests/lean/test_model_resolution.py -v`

Expected: FAIL because `resolve_deepseek_model_class` does not exist.

- [ ] **Step 3: Replace subprocess inspection and registries with direct imports**

```python
SUPPORTED_MODELS: dict[str, tuple[str, str]] = {
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "DeepSeekMTPModel": ("deepseek_mtp", "DeepSeekMTP"),
    "EagleDeepSeekMTPModel": ("deepseek_eagle", "EagleDeepseekV3ForCausalLM"),
    "Eagle3DeepseekV2ForCausalLM": ("deepseek_eagle3", "Eagle3DeepseekV2ForCausalLM"),
    "Eagle3DeepseekV3ForCausalLM": ("deepseek_eagle3", "Eagle3DeepseekV3ForCausalLM"),
}


def resolve_deepseek_model_class(architecture: str) -> type[nn.Module]:
    try:
        module_name, class_name = SUPPORTED_MODELS[architecture]
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture {architecture!r}; expected DeepSeek V2/V3") from exc
    module = importlib.import_module(f"vllm.model_executor.models.{module_name}")
    return getattr(module, class_name)
```

Retain the `ModelRegistry` methods still consumed by `ModelConfig`, but remove out-of-tree registration, Transformers fallback, subprocess inspection, previous-model notices, pooling/task maps, multimodal metadata, and dynamic modules. The explicit lazy import avoids registry/model circular imports without preserving a generic plugin system.

- [ ] **Step 4: Run model/config tests**

Run: `.venv/bin/python -m pytest tests/lean/test_model_resolution.py tests/test_config.py tests/config/test_model_arch_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/registry.py vllm/model_executor/model_loader/utils.py vllm/transformers_utils/config.py tests/lean/test_model_resolution.py tests/test_config.py tests/models/registry.py tests/config/test_model_arch_config.py
git commit -m "Wire DeepSeek model classes directly"
```

### Task 6: Reduce configuration to the six approved groups

**Files:**

- Modify: `vllm/config/__init__.py`
- Modify: `vllm/config/{vllm,model,parallel,load,cache,scheduler,compilation,speculative}.py`
- Modify: `vllm/engine/arg_utils.py`
- Modify: `vllm/sampling_params.py`
- Modify: `vllm/entrypoints/openai/cli_args.py`
- Create: `tests/lean/test_focused_cli.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_sampling_params.py`
- Modify: `tests/entrypoints/openai/test_cli_args.py`
- Delete: unsupported configuration modules listed in the spec.

**Interfaces:**

- Consumes: retained CLI flags and Python constructor kwargs.
- Produces: `VllmConfig` containing only model, parallel, load, cache/scheduler, compilation, speculation, and minimal observability state.

- [ ] **Step 1: Specify accepted and rejected CLI options**

```python
def test_cli_rejects_removed_options(parser) -> None:
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "model", "--enable-lora"])


def test_cli_accepts_focused_parallel_options(parser) -> None:
    args = parser.parse_args([
        "serve", "model", "--tensor-parallel-size", "2",
        "--pipeline-parallel-size", "2", "--data-parallel-size", "2",
        "--enable-expert-parallel",
    ])
    assert (args.tensor_parallel_size, args.pipeline_parallel_size) == (2, 2)
```

Add a sampling contract test that accepts temperature, top-p, top-k, maximum
tokens, stop strings/token IDs, seed, and log probabilities, and rejects LoRA,
structured-output, pooling, multimodal, and tool/reasoning fields.

- [ ] **Step 2: Run CLI/config tests and verify rejected-option failure**

Run: `.venv/bin/python -m pytest tests/lean/test_focused_cli.py tests/entrypoints/openai/test_cli_args.py -v`

Expected: FAIL because removed flags are still registered.

- [ ] **Step 3: Remove unsupported fields at their owning dataclass and parser group**

Keep only the exact groups and fields in the design spec. Replace backend strings with fixed construction. Preserve `enforce_eager` as the explicit diagnostic mode. Preserve only `speculative_config.method in {None, "mtp", "eagle", "eagle3"}`.

Emit one startup log record containing checkpoint source, dtype, TP/PP/DP/EP/PCP/DCP topology, eager/compiled mode, CUDA graph enablement, prefix caching, and speculation mode.

- [ ] **Step 4: Make unknown Python kwargs explicit**

Ensure `EngineArgs` and `LLM` reject unknown or removed kwargs through normal constructor errors; do not retain aliases that silently discard values.

- [ ] **Step 5: Run focused configuration suites**

Run: `.venv/bin/python -m pytest tests/lean/test_focused_cli.py tests/entrypoints/openai/test_cli_args.py tests/test_config.py tests/test_sampling_params.py tests/config -v`

Expected: PASS after deleting tests that exclusively assert removed configuration.

- [ ] **Step 6: Commit**

```bash
git add vllm/config vllm/engine/arg_utils.py vllm/sampling_params.py vllm/entrypoints/openai/cli_args.py tests/lean/test_focused_cli.py tests/test_config.py tests/test_sampling_params.py tests/config tests/entrypoints/openai/test_cli_args.py
git commit -m "Reduce runtime configuration surface"
```

## Phase 3: Focus Online and Offline Entry Points

### Task 7: Reduce the server to five routes

**Files:**

- Modify: `vllm/entrypoints/cli/{main,serve}.py`
- Modify: `vllm/entrypoints/openai/{api_server,cli_args}.py`
- Modify: chat completion, completion, and models router/protocol/serving files.
- Modify: `vllm/entrypoints/serve/instrumentator/{__init__,health,metrics}.py`
- Create: `tests/lean/test_server_routes.py`
- Modify: retained tests under `tests/entrypoints/openai/`.
- Delete: unsupported entrypoint directories and their tests.

**Interfaces:**

- Consumes: `AsyncLLM`, retained request protocols, local DP routing, Prometheus registry.
- Produces: `build_app(args, model_config) -> FastAPI` with exactly five route paths plus FastAPI's disabled documentation internals.

- [ ] **Step 1: Assert the exact route set**

```python
def test_server_has_only_supported_routes(args, model_config) -> None:
    app = build_app(args, model_config=model_config)
    paths = {route.path for route in app.routes if route.path != "/openapi.json"}
    assert paths == {
        "/v1/chat/completions", "/v1/completions", "/v1/models",
        "/health", "/metrics",
    }
```

- [ ] **Step 2: Run the route test and verify extra routes**

Run: `.venv/bin/python -m pytest tests/lean/test_server_routes.py -v`

Expected: FAIL and display the current auxiliary routes.

- [ ] **Step 3: Construct only retained routers**

Remove plugin attachment, SageMaker bootstrap, scale middleware, tool parsers, structured outputs, arbitrary middleware imports, offline docs, CORS customization, response debugging, elastic/fault-tolerance/profile/tokenize/dev routers, Responses, generate/scoring routes, and multiple API-server modes. Retain authentication, request IDs, request logging, exception-to-OpenAI error mapping, health, and Prometheus metrics.

- [ ] **Step 4: Reduce the CLI to one server process and local DP engines**

`ServeSubcommand.cmd()` must call `run_server(args)` directly for DP=1 and the focused local DP supervisor for DP>1. Remove gRPC, Rust frontend, headless, external/hybrid/multi-port load balancing, multiple API server processes, and usage telemetry.

- [ ] **Step 5: Run protocol and route suites**

Run: `.venv/bin/python -m pytest tests/lean/test_server_routes.py tests/entrypoints/openai/chat_completion tests/entrypoints/openai/completion tests/entrypoints/openai/models tests/entrypoints/serve/instrumentator -v`

Expected: PASS after removing tests for tools, multimodal inputs, prompt embeddings, token-in/token-out, Responses, and auxiliary routes.

- [ ] **Step 6: Commit**

```bash
git add vllm/entrypoints tests/lean/test_server_routes.py tests/entrypoints
git commit -m "Focus the OpenAI serving surface"
```

### Task 8: Simplify local data-parallel routing

**Files:**

- Modify: `vllm/entrypoints/openai/dp_supervisor.py`
- Modify: `vllm/v1/engine/{coordinator,core_client,utils}.py`
- Modify: `tests/entrypoints/openai/test_dp_supervisor.py`
- Modify: `tests/v1/distributed/test_async_llm_dp.py`

**Interfaces:**

- Consumes: one HTTP process, `data_parallel_size`, and local engine addresses.
- Produces: `LocalDPRouter` that assigns each request to one healthy local DP engine and one `CoreEngineProcManager` per replica.

- [ ] **Step 1: Characterize local round-robin and failure behavior**

```python
def test_local_dp_router_round_robins_healthy_engines() -> None:
    router = LocalDPRouter(["engine-0", "engine-1"])
    assert [router.next_engine() for _ in range(4)] == [
        "engine-0", "engine-1", "engine-0", "engine-1",
    ]


def test_local_dp_router_fails_when_engine_dies() -> None:
    router = LocalDPRouter(["engine-0"])
    router.mark_failed("engine-0")
    with pytest.raises(RuntimeError, match="local DP engine failed"):
        router.next_engine()
```

- [ ] **Step 2: Run DP tests and verify the focused router is absent**

Run: `.venv/bin/python -m pytest tests/entrypoints/openai/test_dp_supervisor.py tests/v1/distributed/test_async_llm_dp.py -v`

Expected: FAIL for the new `LocalDPRouter` behavior.

- [ ] **Step 3: Delete remote/external modes and implement the local router**

Retain local subprocess launch, engine health, request assignment, metrics aggregation, and coordinated shutdown. Remove remote addresses, per-node ranks, external and hybrid load balancing, multi-port serving, Ray actors, Kubernetes assumptions, and engine reconnection.

- [ ] **Step 4: Run DP suites**

Run: `.venv/bin/python -m pytest tests/entrypoints/openai/test_dp_supervisor.py tests/v1/distributed/test_async_llm_dp.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/entrypoints/openai/dp_supervisor.py vllm/v1/engine tests/entrypoints/openai/test_dp_supervisor.py tests/v1/distributed/test_async_llm_dp.py
git commit -m "Keep local data parallel routing"
```

### Task 9: Keep offline generate and chat on the same V1 engine

**Files:**

- Modify: `vllm/entrypoints/llm.py`
- Modify: `vllm/entrypoints/offline_utils.py`
- Modify: `vllm/entrypoints/chat_utils.py`
- Modify: `vllm/__init__.py`
- Modify: `tests/entrypoints/llm/test_chat.py`
- Create: `tests/lean/test_offline_contract.py`

**Interfaces:**

- Consumes: `LLMEngine`, retained tokenizer/chat template, `SamplingParams`.
- Produces: `LLM.generate()` and `LLM.chat()` only, both delegating to the V1 engine.

- [ ] **Step 1: Test the method surface without loading a model**

```python
def test_llm_exposes_only_retained_inference_methods() -> None:
    assert hasattr(LLM, "generate")
    assert hasattr(LLM, "chat")
    assert not hasattr(LLM, "beam_search")
    assert not hasattr(LLM, "encode")
    assert not hasattr(LLM, "score")
```

- [ ] **Step 2: Run and verify the current extra surface**

Run: `.venv/bin/python -m pytest tests/lean/test_offline_contract.py -v`

Expected: FAIL because beam-search or auxiliary methods remain.

- [ ] **Step 3: Remove offline mixins and unsupported chat features**

Keep text conversations, Hugging Face chat-template rendering, batch cleanup on validation failure, tokenization, generation, and detokenization. Remove beam, pooling, multimodal content, tools, reasoning-template controls, prompt embeddings, weight transfer, sleep, and profiling methods.

- [ ] **Step 4: Retarget tests to DeepSeek**

Use `deepseek-ai/DeepSeek-V2-Lite-Chat` for BF16 chat tests and `ZixiQi/DeepSeek-V3-4layers-MTP-FP8` for lightweight FP8 generation tests. Mark network/model tests with the existing model-download and Hopper markers.

- [ ] **Step 5: Run offline contract and chat tests**

Run: `.venv/bin/python -m pytest tests/lean/test_offline_contract.py tests/entrypoints/llm/test_chat.py -v`

Expected: PASS on a configured Hopper runner; CPU collection must also succeed.

- [ ] **Step 6: Commit**

```bash
git add vllm/__init__.py vllm/entrypoints/llm.py vllm/entrypoints/offline_utils.py vllm/entrypoints/chat_utils.py tests/lean/test_offline_contract.py tests/entrypoints/llm/test_chat.py
git commit -m "Focus the offline LLM API"
```

## Phase 4: Specialize Speculation and Hopper Execution

### Task 10: Reduce speculative decoding to MTP and EAGLE

**Files:**

- Modify: `vllm/config/speculative.py`
- Modify: `vllm/v1/spec_decode/llm_base_proposer.py`
- Modify: `vllm/v1/worker/gpu/spec_decode/`
- Modify: `tests/v1/e2e/spec_decode/mtp/test_mtp.py`
- Modify: `tests/v1/e2e/spec_decode/eagle/test_eagle_correctness.py`
- Modify: `tests/v1/spec_decode/test_speculators_eagle3.py`
- Delete: generic proposer modules and tests.

**Interfaces:**

- Consumes: target DeepSeek model, embedded MTP layers or EAGLE checkpoint.
- Produces: speculation method `None | "mtp" | "eagle" | "eagle3"` and retained draft/verify execution.

- [ ] **Step 1: Assert accepted and rejected methods**

```python
@pytest.mark.parametrize("method", ["mtp", "eagle", "eagle3"])
def test_supported_speculative_methods(method: str) -> None:
    assert SpeculativeConfig(method=method, num_speculative_tokens=1).method == method


@pytest.mark.parametrize("method", ["ngram", "suffix", "medusa", "draft_model"])
def test_rejects_generic_speculative_methods(method: str) -> None:
    with pytest.raises(ValueError, match="MTP or EAGLE"):
        SpeculativeConfig(method=method, num_speculative_tokens=1)
```

- [ ] **Step 2: Run and verify generic methods are still accepted**

Run: `.venv/bin/python -m pytest tests/config/test_speculative_draft_hf_overrides.py tests/v1/spec_decode -v`

Expected: FAIL against the focused method contract.

- [ ] **Step 3: Delete generic proposer construction**

Retain shared metadata, metrics, vocabulary mapping, MTP hidden-state extraction, EAGLE draft attention, EAGLE3 utilities, rejection sampling, and target verification. Delete n-gram CPU/GPU, suffix, custom-class, Medusa, DFlash, Gemma4, Step3.5, generic draft models, and dynamic proposer code.

- [ ] **Step 4: Retarget correctness cases**

Keep `ZixiQi/DeepSeek-V3-4layers-MTP-FP8` for MTP and the `eagle618/deepseek-v3-random` plus `eagle618/eagle-deepseek-v3-random` pair for EAGLE. Keep EAGLE3 architecture/unit tests without introducing a non-DeepSeek target checkpoint.

- [ ] **Step 5: Run speculation tests**

Run: `.venv/bin/python -m pytest tests/v1/spec_decode tests/v1/e2e/spec_decode/mtp/test_mtp.py tests/v1/e2e/spec_decode/eagle/test_eagle_correctness.py -v`

Expected: PASS on Hopper; CPU collection succeeds.

- [ ] **Step 6: Commit**

```bash
git add vllm/config/speculative.py vllm/v1/spec_decode vllm/v1/worker/gpu/spec_decode tests/v1/spec_decode tests/v1/e2e/spec_decode
git commit -m "Keep DeepSeek MTP and EAGLE decoding"
```

### Task 11: Directly select Hopper MLA attention

**Files:**

- Modify: `vllm/model_executor/layers/attention/mla_attention.py`
- Modify: `vllm/v1/attention/backends/registry.py`
- Modify: `vllm/v1/attention/backends/mla/prefill/selector.py`
- Retain and simplify: FlashAttention prefill and FlashMLA decode files.
- Modify: `tests/v1/attention/test_mla_backends.py`
- Modify: `tests/kernels/attention/test_flashmla.py`
- Delete: other attention backends and selection tests.

**Interfaces:**

- Consumes: DeepSeek MLA metadata and Hopper capability invariant.
- Produces: direct FlashAttention prefill and FlashMLA decode construction.

- [ ] **Step 1: Test deterministic backend selection**

```python
def test_hopper_mla_backend_is_fixed() -> None:
    assert select_mla_prefill_backend() is FlashAttnPrefillBackend
    assert select_mla_decode_backend() is FlashMLABackend
```

- [ ] **Step 2: Verify current selector exposes alternatives**

Run: `.venv/bin/python -m pytest tests/v1/attention/test_mla_prefill_selector.py tests/v1/attention/test_mla_backends.py -v`

Expected: FAIL against the fixed Hopper choice.

- [ ] **Step 3: Remove attention registries and fallback branches**

Delete generic MHA, Triton MLA fallback, FlashInfer MLA, CUTLASS MLA, sparse MLA, Tokenspeed, TRT-LLM ragged, CPU/ROCm/XPU branches, environment overrides, and backend-name CLI/config fields. Preserve the shared MLA metadata and paged-cache operations consumed by FlashAttention/FlashMLA.

- [ ] **Step 4: Run MLA kernel and integration tests**

Run: `.venv/bin/python -m pytest tests/v1/attention/test_mla_backends.py tests/v1/attention/test_mla_context_chunks.py tests/kernels/attention/test_flashmla.py tests/model_executor/layers/test_mla_short_prefill_indexer.py -v`

Expected: PASS on H100/H200.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/layers/attention vllm/v1/attention tests/v1/attention tests/kernels/attention tests/model_executor/layers/test_mla_short_prefill_indexer.py
git commit -m "Specialize MLA attention for Hopper"
```

### Task 12: Directly select Hopper BF16 and FP8 MoE kernels

**Files:**

- Modify: `vllm/model_executor/layers/fused_moe/{layer,config,all2all_utils}.py`
- Create: `vllm/model_executor/layers/fused_moe/experts/hopper_bf16.py`
- Modify: `vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py`
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Modify: `vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py`
- Modify: `vllm/model_executor/warmup/deep_gemm_warmup.py`
- Modify: focused tests under `tests/kernels/moe/` and `tests/kernels/quantization/`.
- Delete: non-Hopper and non-BF16/FP8 expert implementations and tuning files.

**Interfaces:**

- Consumes: DeepSeek MoE configuration, dtype, static EP group.
- Produces: one BF16 Hopper expert path, one DeepGEMM FP8 expert path, and all-gather/reduce-scatter EP preparation/finalization.

- [ ] **Step 1: Test fixed dtype dispatch**

```python
def test_deepseek_moe_kernel_for_dtype() -> None:
    assert select_deepseek_moe_kernel(torch.bfloat16) is HopperBF16Experts
    assert select_deepseek_moe_kernel(torch.float8_e4m3fn) is DeepGemmExperts


def test_rejects_other_moe_dtype() -> None:
    with pytest.raises(ValueError, match="BF16 or FP8"):
        select_deepseek_moe_kernel(torch.float16)
```

- [ ] **Step 2: Run selector tests and verify current generic dispatch**

Run: `.venv/bin/python -m pytest tests/kernels/moe/test_moe_layer.py tests/kernels/moe/test_block_fp8.py -v`

Expected: FAIL against the direct two-path contract.

- [ ] **Step 3: Collapse expert construction and tuning data**

Extract `HopperBF16Experts` from the unquantized subset of `TritonExperts`, without the LoRA mixin or generic quantization branches. Keep DeepSeek grouped top-k routing, grouped GEMM, shared experts, FP8 block quantization, DeepGEMM packing/JIT warmup, and H100/H200 configurations whose shapes match DeepSeek V2/V3. Delete integer, MX, NVFP4, Marlin, CUTLASS alternatives, modular backend selection, model-specific routers, non-Hopper tuning JSON, and alternate EP prepare/finalize implementations.

- [ ] **Step 4: Run BF16/FP8 MoE tests**

Run: `.venv/bin/python -m pytest tests/kernels/moe/test_moe_layer.py tests/kernels/moe/test_block_fp8.py tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py tests/kernels/quantization/test_fp8_quant.py -v`

Expected: PASS on Hopper.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/layers/fused_moe vllm/model_executor/layers/quantization vllm/model_executor/kernels/linear vllm/model_executor/warmup tests/kernels/moe tests/kernels/quantization
git commit -m "Specialize DeepSeek MoE for Hopper"
```

### Task 13: Retain only compilation and CUDA graph paths exercised by DeepSeek

**Files:**

- Modify: `vllm/config/compilation.py`
- Modify: `vllm/compilation/`
- Modify: `vllm/v1/worker/gpu/cudagraph_utils.py`
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Create: `tests/lean/test_optimized_parity.py`
- Modify: focused tests under `tests/compile/` and `tests/v1/cudagraph/`.

**Interfaces:**

- Consumes: DeepSeek model graph, scheduler batch shapes, Hopper kernels.
- Produces: eager diagnostic mode and compiled CUDA-graph mode with explicit failure.

- [ ] **Step 1: Add eager/compiled parity characterization**

```python
@pytest.mark.parametrize("dtype", ["bfloat16", "fp8"])
def test_optimized_matches_eager(deepseek_runner, dtype: str) -> None:
    eager = deepseek_runner(dtype=dtype, enforce_eager=True).generate(DETERMINISTIC_PROMPTS)
    optimized = deepseek_runner(dtype=dtype, enforce_eager=False).generate(DETERMINISTIC_PROMPTS)
    assert_token_outputs_equal(eager, optimized)
```

- [ ] **Step 2: Run the existing DeepSeek compile/cudagraph subset**

Run: `.venv/bin/python -m pytest tests/compile/passes/test_fuse_mla_dual_rms_norm.py tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py tests/v1/cudagraph -v`

Expected: record the baseline before deleting passes.

- [ ] **Step 3: Delete passes and graph modes not reached by DeepSeek**

Retain bytecode/inductor integration, piecewise/full graph modes actually selected by the retained runner, MLA/RMS/quant fusion passes, graph memory pool, shape capture, and DeepGEMM warmup. Remove diffusion, multimodal, non-DeepSeek model, non-Hopper, alternate quantization, and deleted collective fusions.

- [ ] **Step 4: Make optimized failure explicit**

Remove silent fallback from compiled/graph mode to eager. Preserve `enforce_eager=True` as the only deliberate eager selection.

- [ ] **Step 5: Run compile, graph, and parity tests**

Run: `.venv/bin/python -m pytest tests/compile tests/v1/cudagraph tests/lean/test_optimized_parity.py -v`

Expected: PASS on Hopper; CPU-safe tests collect successfully.

- [ ] **Step 6: Commit**

```bash
git add vllm/config/compilation.py vllm/compilation vllm/v1/worker tests/compile tests/v1/cudagraph tests/lean/test_optimized_parity.py
git commit -m "Focus compilation and CUDA graphs on DeepSeek"
```

## Phase 5: Simplify Every Local Parallel Mode

### Task 14: Collapse process groups to local TP, PP, DP, PCP, and DCP

**Files:**

- Modify: `vllm/distributed/parallel_state.py`
- Modify: `vllm/distributed/communication_op.py`
- Modify: `vllm/distributed/device_communicators/{base_device_communicator,cuda_communicator,shm_broadcast}.py`
- Modify: `vllm/v1/worker/{cp_utils,dp_utils}.py`
- Modify: `vllm/v1/worker/gpu/{dp_utils,pp_utils}.py`
- Modify: `tests/distributed/test_context_parallel.py`
- Create: `tests/lean/test_parallel_topology.py`

**Interfaces:**

- Consumes: local rank plus TP/PP/DP/PCP/DCP sizes.
- Produces: deterministic local rank groups and NCCL/Gloo coordinators.

- [ ] **Step 1: Test one concrete combined topology**

```python
def test_combined_local_rank_groups() -> None:
    topology = build_local_topology(tp=2, pp=2, dp=2, pcp=1, dcp=2)
    assert topology.local_gpu_count == 8
    assert topology.tp_groups == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert topology.pp_groups == ((0, 2), (1, 3), (4, 6), (5, 7))
    assert all(len(group) == 2 for group in topology.dp_groups)
    assert all(len(group) == 2 for group in topology.dcp_groups)
```

- [ ] **Step 2: Run and verify the topology helper is absent**

Run: `.venv/bin/python -m pytest tests/lean/test_parallel_topology.py -v`

Expected: FAIL because `build_local_topology` does not exist.

- [ ] **Step 3: Extract pure topology algebra from process-group creation**

Implement `build_local_topology()` as a frozen description used by both validation and `initialize_model_parallel()`. Remove stateless groups, multi-node rank offsets, external stores, network-device routing, elastic groups, Ray communicators, and reconfiguration.

- [ ] **Step 4: Keep only NCCL and required Gloo groups**

Device tensor collectives use NCCL; object/control coordination uses Gloo only where the executor or DP router requires it. Delete alternate device communicators and transport selection.

- [ ] **Step 5: Run topology and distributed unit tests**

Run: `.venv/bin/python -m pytest tests/lean/test_parallel_topology.py tests/distributed/test_context_parallel.py tests/v1/engine/test_parallel_sampling.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/distributed vllm/v1/worker/cp_utils.py vllm/v1/worker/dp_utils.py vllm/v1/worker/gpu tests/lean/test_parallel_topology.py tests/distributed/test_context_parallel.py tests/v1/engine/test_parallel_sampling.py
git commit -m "Simplify local model parallel groups"
```

### Task 15: Keep static expert parallelism with one all-to-all

**Files:**

- Modify: `vllm/distributed/device_communicators/all2all.py`
- Modify: `vllm/model_executor/layers/fused_moe/all2all_utils.py`
- Modify: `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`
- Modify: `vllm/model_executor/model_loader/ep_weight_filter.py`
- Create: `tests/lean/test_static_expert_parallel.py`
- Modify: focused EP tests under `tests/distributed/` and `tests/kernels/moe/`.

**Interfaces:**

- Consumes: static linear expert placement and local EP group.
- Produces: `compute_local_expert_ids(num_experts: int, ep_size: int, ep_rank: int) -> set[int] | None`, all-gather inputs, local expert execution, and reduce-scatter outputs.

- [ ] **Step 1: Test static expert ownership**

```python
def test_linear_expert_placement() -> None:
    assert compute_local_expert_ids(num_experts=8, ep_size=2, ep_rank=0) == {0, 1, 2, 3}
    assert compute_local_expert_ids(num_experts=8, ep_size=2, ep_rank=1) == {4, 5, 6, 7}
```

- [ ] **Step 2: Run the focused test and record generic backend behavior**

Run: `.venv/bin/python -m pytest tests/lean/test_static_expert_parallel.py -v`

Expected: FAIL until the focused helper and direct all-to-all are exposed.

- [ ] **Step 3: Remove backend and placement selection**

Keep linear placement, EP-aware safetensors filtering, all-gather/reduce-scatter preparation/finalization, and DeepSeek shared experts. Delete round-robin placement, EPLB, redundant experts, DeepEP, NIXL, MoRI, FlashInfer all-to-all, naive debug backend selection, and dynamic routing capture.

- [ ] **Step 4: Run EP tests**

Run: `.venv/bin/python -m pytest tests/lean/test_static_expert_parallel.py tests/model_executor/model_loader/test_ep_weight_filter.py tests/kernels/moe/test_moe_layer.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/distributed/device_communicators/all2all.py vllm/model_executor/layers/fused_moe vllm/model_executor/model_loader/ep_weight_filter.py tests/lean/test_static_expert_parallel.py tests/model_executor/model_loader/test_ep_weight_filter.py tests/kernels/moe/test_moe_layer.py
git commit -m "Keep static local expert parallelism"
```

### Task 16: Add focused multi-GPU topology smoke tests

**Files:**

- Create: `tests/lean/test_multigpu_topologies.py`
- Create: `tests/lean/topology_cases.py`
- Modify: `tests/conftest.py`

**Interfaces:**

- Consumes: focused `LLM`, a small DeepSeek safetensors fixture, local parallel flags.
- Produces: one-mode and representative-combination GPU validation.

- [ ] **Step 1: Define the finite topology matrix**

```python
TOPOLOGY_CASES = (
    pytest.param(dict(tp=2), id="tp2"),
    pytest.param(dict(pp=2), id="pp2"),
    pytest.param(dict(dp=2), id="dp2"),
    pytest.param(dict(tp=2, ep=True), id="tp2-ep"),
    pytest.param(dict(tp=2, dcp=2), id="tp2-dcp2"),
    pytest.param(dict(pcp=2), id="pcp2"),
    pytest.param(dict(tp=2, pp=2, dp=2), id="tp2-pp2-dp2"),
)
```

- [ ] **Step 2: Add one-token deterministic generation per case**

Use the same prompt, seed, and greedy `SamplingParams(max_tokens=1, temperature=0)` for each topology. Assert one output, one generated token, and clean worker shutdown.

- [ ] **Step 3: Run on an 8x Hopper host**

Run: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -m pytest tests/lean/test_multigpu_topologies.py -v`

Expected: PASS; cases automatically skip only when their declared GPU count exceeds visible SM90 devices.

- [ ] **Step 4: Commit**

```bash
git add tests/lean/test_multigpu_topologies.py tests/lean/topology_cases.py tests/conftest.py
git commit -m "Cover retained local parallel topologies"
```

## Phase 6: Delete Unsupported Runtime and Repository Breadth

### Task 17: Focus the V1 scheduler and GPU runner on text generation

**Files:**

- Modify: `vllm/v1/core/sched/scheduler.py`
- Modify: `vllm/v1/core/kv_cache_manager.py`
- Modify: `vllm/v1/engine/{core,input_processor,output_processor}.py`
- Modify: `vllm/v1/request.py`
- Modify: `vllm/v1/outputs.py`
- Modify: `vllm/v1/worker/{gpu_model_runner,gpu_input_batch}.py`
- Modify: `tests/v1/core/test_scheduler.py`
- Modify: `tests/v1/core/test_prefix_caching.py`
- Create: `tests/lean/test_generation_engine_contract.py`

**Interfaces:**

- Consumes: text token requests, retained sampling fields, cache/scheduler configuration, and optional MTP/EAGLE metadata.
- Produces: one generation-only scheduler/model-runner path with continuous batching, paged KV cache, prefix caching, parallel sampling, and speculative verification.

- [ ] **Step 1: Assert the retained task and request state**

```python
def test_engine_supports_generation_only(engine_config) -> None:
    engine = EngineCore.__new__(EngineCore)
    engine.vllm_config = engine_config
    engine.model_executor = SimpleNamespace(supported_tasks=("generate",))
    assert engine.get_supported_tasks() == ("generate",)


def test_request_has_no_removed_feature_state() -> None:
    request = Request("request-0", [1, 2], SamplingParams(max_tokens=1))
    assert not hasattr(request, "lora_request")
    assert not hasattr(request, "structured_output_request")
    assert not hasattr(request, "mm_features")
    assert not hasattr(request, "pooling_params")
```

- [ ] **Step 2: Run the engine contract and record extra state**

Run: `.venv/bin/python -m pytest tests/lean/test_generation_engine_contract.py -v`

Expected: FAIL because retained request/engine objects still expose removed feature state.

- [ ] **Step 3: Remove guarded branches from core data flow**

Remove pooling, multimodal encoder/cache, structured grammar, LoRA, KV/EC connector, offload, weight-update, sleep, mamba/GDN, tracing, and fault-tolerance state from requests, scheduler outputs, engine core, GPU input batches, and the model runner. Preserve text prompt tokens, block tables, scheduler events, prefix hashes, sampled/logprob outputs, MTP/EAGLE draft IDs, PP intermediate tensors, and local DP coordination.

- [ ] **Step 4: Split the runner only at stable responsibilities**

Keep `gpu_model_runner.py` as the orchestration boundary, but move retained CUDA-graph shape bookkeeping to `vllm/v1/worker/gpu/cudagraph_utils.py` and retained speculative helpers to `vllm/v1/worker/gpu/spec_decode/`. Do not introduce another generic runner registry.

- [ ] **Step 5: Run scheduler, cache, engine, and runner tests**

Run: `.venv/bin/python -m pytest tests/lean/test_generation_engine_contract.py tests/v1/core/test_scheduler.py tests/v1/core/test_prefix_caching.py tests/v1/core/test_kv_cache_utils.py tests/v1/engine -v`

Expected: PASS; no retained test imports a removed request state.

- [ ] **Step 6: Commit**

```bash
git add vllm/v1/core vllm/v1/engine vllm/v1/request.py vllm/v1/outputs.py vllm/v1/worker tests/lean/test_generation_engine_contract.py tests/v1/core tests/v1/engine
git commit -m "Focus the V1 engine on text generation"
```

### Task 18: Remove unsupported runtime subsystems and compatibility stubs

**Files:**

- Delete: `vllm/lora/`, `vllm/multimodal/`, `vllm/plugins/`, `vllm/reasoning/`, `vllm/tool_parsers/`, `vllm/v1/pool/`, `vllm/v1/structured_output/`, KV/EC transfer and offload directories, weight transfer, elastic/fault-tolerance code, and corresponding tests.
- Modify: all retained callers found by the static guard.
- Create: `tools/check_lean_tree.py`
- Create: `tests/lean/test_tree_contract.py`

**Interfaces:**

- Consumes: approved root paths and banned package/backend lists.
- Produces: `check_lean_tree(repo_root: Path) -> list[str]` returning violations.

- [ ] **Step 1: Write the static contract before deletion**

```python
BANNED_PATHS = (
    "vllm/lora", "vllm/multimodal", "vllm/plugins", "vllm/reasoning",
    "vllm/tool_parsers", "vllm/v1/pool", "vllm/v1/structured_output",
    "vllm/v1/kv_offload", "vllm/v1/simple_kv_offload",
    "vllm/model_executor/layers/mamba", "vllm/third_party/flash_linear_attention",
)
BANNED_TERMS = (
    "ray", "external_launcher", "nixl", "mooncake", "lmcache",
    "structured_output", "enable_lora", "multimodal",
)
```

The checker scans tracked Python/build files, reports file and line for banned imports/qualnames/backend registrations, checks package `__init__.py` boundaries, and rejects modules containing `No-op stub`.

- [ ] **Step 2: Run and record violations**

Run: `.venv/bin/python -m pytest tests/lean/test_tree_contract.py -v`

Expected: FAIL with the remaining unsupported references.

- [ ] **Step 3: Remove one subsystem family at a time**

Delete in this order, running the public contract after each family: LoRA/multimodal/pooling; structured output/tools/reasoning; KV/EC transfer and offload; weight transfer/sleep; plugins/dynamic registration; elastic/fault-tolerance/tracing/usage telemetry/profilers. Fold the few retained utility types into their sole caller instead of leaving package shells.

- [ ] **Step 4: Run import and core behavior suites**

Run: `.venv/bin/python -m pytest tests/lean tests/v1/core tests/v1/engine tests/entrypoints/llm tests/entrypoints/openai -v`

Expected: PASS; the static checker reports no banned paths, terms, unresolved qualnames, or stubs.

- [ ] **Step 5: Commit**

```bash
git add vllm tests tools/check_lean_tree.py
git commit -m "Remove unsupported runtime subsystems"
```

### Task 19: Prune CUDA build inputs and dependencies to Hopper DeepSeek

**Files:**

- Modify: `CMakeLists.txt`, `setup.py`, `pyproject.toml`, `MANIFEST.in`.
- Modify: `cmake/`, `requirements/`, `.pre-commit-config.yaml`, `.buildkite/ci_config.yaml`.
- Delete: unused `csrc/` sources, Docker files, tools, and dependency groups.
- Modify: `tests/tools/test_docker_build_metadata_args.py` and focused build tests.

**Interfaces:**

- Consumes: retained Python native-op registrations and SM90 source list.
- Produces: a CUDA SM90 build containing only native operations reached by the focused runtime.

- [ ] **Step 1: Generate a retained-op manifest from imports and registrations**

Add `RETAINED_NATIVE_OPS` to `tools/check_lean_tree.py`; each entry names the Python call site, registration name, and source file. The checker fails when a retained registration has no source or an unlisted native source remains in an extension target.

- [ ] **Step 2: Run the manifest check before pruning**

Run: `.venv/bin/python tools/check_lean_tree.py --native-ops`

Expected: FAIL with unused extension sources and registrations.

- [ ] **Step 3: Restrict build architecture and sources**

Require SM90 in CMake configuration; retain FlashAttention, FlashMLA, DeepGEMM, paged-cache, sampling, FP8, MoE, and collective sources referenced by the manifest. Remove other GPU-generation, model, quantization, multimodal, mamba, sparse-attention, and alternate-backend sources.

- [ ] **Step 4: Remove dependency breadth**

Keep only packages imported by the retained tree and its focused tests. Remove Ray, tensorizer, RunAI, ModelScope, LoRA, multimodal media, structured-output, alternate quantization, remote transport, and documentation-generator dependencies.

- [ ] **Step 5: Configure and build on Hopper**

Run: `VLLM_USE_PRECOMPILED=0 uv pip install -e . --torch-backend=auto`

Expected: build succeeds and logs only SM90 CUDA targets.

- [ ] **Step 6: Run native-op and build tests**

Run: `.venv/bin/python tools/check_lean_tree.py --native-ops`

Run: `.venv/bin/python -m pytest tests/tools/test_docker_build_metadata_args.py tests/kernels/attention/test_flashmla.py tests/kernels/moe/test_block_fp8.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt setup.py pyproject.toml MANIFEST.in cmake csrc requirements docker tools .pre-commit-config.yaml .buildkite/ci_config.yaml tests/tools
git commit -m "Focus the build on Hopper DeepSeek"
```

### Task 20: Replace the test, example, and documentation breadth

**Files:**

- Keep and focus: `tests/lean/`, selected V1 scheduler/cache, Hopper kernel, API, offline, and spec-decode tests.
- Create: `examples/lean/offline_generate.py`
- Create: `examples/lean/serve.sh`
- Create: `docs/lean/{architecture,reading-order,parallelism,hopper-kernels}.md`
- Create: `tests/evals/gsm8k/configs/deepseek/{bf16,fp8,mtp,eagle}.yaml`
- Modify: `README.md` and documentation navigation.
- Keep unchanged: `SECURITY.md`, `AGENTS.md`, `docs/usage/security.md`, `docs/contributing/vulnerability_management.md`, and `docs/contributing/editing-agent-instructions.md`.
- Delete: unrelated tests, examples, benchmarks, and generated docs.

**Interfaces:**

- Consumes: final focused tree.
- Produces: an accurate learning path and a 25k-45k line focused Python test suite.

- [ ] **Step 1: Write examples that use only approved interfaces**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite-Chat", tensor_parallel_size=2)
outputs = llm.generate(
    ["Explain continuous batching in one paragraph."],
    SamplingParams(temperature=0, max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

`serve.sh` invokes `vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --tensor-parallel-size 2` and shows one streaming chat-completion request.

- [ ] **Step 2: Write the four focused guides**

`architecture.md` traces online/offline requests to tokens; `reading-order.md` gives exact files in study order; `parallelism.md` shows rank formulas and the retained topology cases; `hopper-kernels.md` maps FlashAttention, FlashMLA, DeepGEMM, FP8, CUDA graph, and native-op call sites.

The four GSM8K configurations use `deepseek-ai/DeepSeek-V2-Lite-Chat` for BF16, `ZixiQi/DeepSeek-V3-4layers-MTP-FP8` for FP8 and MTP, and the `eagle618/deepseek-v3-random` target with `eagle618/eagle-deepseek-v3-random` for EAGLE. Each file fixes the same prompt set, seed, maximum token count, and eager/optimized comparison fields.

- [ ] **Step 3: Prune tests only after coverage mapping**

Create a table in `docs/lean/architecture.md` mapping each acceptance criterion to at least one retained test. Remove tests only when they target a deleted feature or duplicate a cheaper retained behavior test.

- [ ] **Step 4: Enforce source and test budgets without hiding necessary code**

Extend `tools/check_lean_tree.py --budgets` to report Python runtime and test LOC. Fail above 170,000 runtime lines or 45,000 test lines; report but do not fail below the lower design targets.

- [ ] **Step 5: Run documentation examples as smoke tests**

Run: `.venv/bin/python examples/lean/offline_generate.py`

Run: `bash -n examples/lean/serve.sh`

Expected: offline generation succeeds on Hopper; shell syntax passes.

- [ ] **Step 6: Commit**

```bash
git add README.md docs examples tests tools/check_lean_tree.py
git commit -m "Add the focused DeepSeek study repository"
```

## Phase 7: Final Verification and Handoff

### Task 21: Run full focused verification, evaluations, and review

**Files:**

- Modify only files required to fix failures found by these commands.
- Record results in the final handoff; do not add transient logs to the repository.

**Interfaces:**

- Consumes: completed focused repository.
- Produces: evidence that every acceptance criterion is satisfied.

- [ ] **Step 1: Run static and formatting verification**

Run: `.venv/bin/python tools/check_lean_tree.py --all`

Run: `pre-commit run --all-files`

Run: `pre-commit run mypy-3.12 --all-files --hook-stage manual`

Run: `git diff --check main...HEAD`

Expected: all commands pass; no banned imports, unresolved qualnames, stubs, whitespace errors, or budget violations.

- [ ] **Step 2: Run CPU-safe focused tests**

Run: `.venv/bin/python -m pytest tests/lean tests/config tests/v1/core tests/v1/engine tests/entrypoints/openai tests/entrypoints/llm -v`

Expected: PASS or explicit Hopper skips only for tests that execute CUDA kernels/models.

- [ ] **Step 3: Run Hopper kernel and optimized parity tests**

Run: `.venv/bin/python -m pytest tests/kernels/attention tests/kernels/moe tests/kernels/quantization tests/compile tests/v1/cudagraph -v`

Expected: PASS on H100/H200.

- [ ] **Step 4: Run multi-GPU topology tests**

Run: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -m pytest tests/lean/test_multigpu_topologies.py -v`

Expected: PASS on an 8x Hopper host.

- [ ] **Step 5: Run online/offline base, MTP, and EAGLE end-to-end tests**

Run: `.venv/bin/python -m pytest tests/entrypoints/llm tests/entrypoints/openai/chat_completion tests/entrypoints/openai/completion tests/v1/e2e/spec_decode/mtp/test_mtp.py tests/v1/e2e/spec_decode/eagle/test_eagle_correctness.py -v`

Expected: PASS on Hopper.

- [ ] **Step 6: Run model evaluation and serving performance checks**

Run the retained GSM8K evaluation for BF16, FP8, MTP, and EAGLE using the focused configuration files under `tests/evals/gsm8k/configs/deepseek/`. Record accuracy, exact-match comparison where deterministic, speculative acceptance rate, output tokens/second, time to first token, and inter-token latency. Compare optimized against eager and compare the focused branch against commit `f35a42ef7a` using the same model, prompts, topology, and GPU clocks.

Expected: no agreed quality regression; optimized mode is not slower than eager for the measured serving workload. Any regression blocks completion and is diagnosed before proceeding.

- [ ] **Step 7: Request code review**

Invoke `superpowers:requesting-code-review` over the complete diff. Address findings using `superpowers:receiving-code-review`, rerun the affected commands, and then run `superpowers:verification-before-completion` before claiming completion.

- [ ] **Step 8: Commit final verification fixes**

```bash
git add -u
git commit -m "Complete focused DeepSeek verification"
```

Skip this commit when verification required no code changes.

## Review Checkpoints

- After Phase 1: local multi-GPU executor imports, starts, and shuts down cleanly.
- After Phase 2: only DeepSeek safetensors models/configuration resolve.
- After Phase 3: the five online routes and offline `generate`/`chat` work.
- After Phase 4: Hopper base, MTP, EAGLE, BF16, FP8, compile, and CUDA graphs work.
- After Phase 5: every retained local parallel mode has focused coverage.
- After Phase 6: static closure passes and repository budgets are met.
- After Phase 7: all verification, evaluation, performance, and review evidence is recorded.
