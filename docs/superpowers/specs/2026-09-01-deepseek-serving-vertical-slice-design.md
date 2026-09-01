# DeepSeek Serving Vertical Slice Design

Date: 2026-09-01

## Status

Approved in conversation for specification. Implementation has not started.

## Context

The `lean-vllm` branch has already removed most model architectures and many
optional subsystems, but the retained repository is still too large and
indirect to study from end to end. It contains approximately 318,000 lines of
Python runtime code and 276,000 lines of Python tests. Generic registries,
option-heavy configuration, compatibility stubs, alternate deployment paths,
and hardware fallbacks obscure the request-to-token path that the repository
is intended to teach.

This design turns the branch into an explicit production-oriented vertical
slice for DeepSeek inference. It preserves optimized online and offline
inference, all logical parallel modes, and the V1 engine, while restricting
hardware, models, checkpoint format, serving APIs, and communication backends.

The result is a focused fork for study and experimentation. It is not intended
to remain API-compatible with general-purpose upstream vLLM.

## Goals

- Make the complete online and offline DeepSeek inference path practical to
  read from entrypoint to CUDA kernels.
- Serve DeepSeek V2 and V3 on a single Hopper host using local NVIDIA GPUs.
- Preserve optimized execution: continuous batching, paged KV cache, prefix
  caching, `torch.compile`, CUDA graphs, kernel warmup, BF16, and FP8.
- Preserve MTP, EAGLE, and EAGLE3 speculative decoding.
- Preserve tensor, pipeline, local data, expert, prefill-context, and
  decode-context parallelism.
- Retain only one local implementation of each parallel and communication
  mechanism.
- Support both a minimal OpenAI-compatible server and the offline `LLM` API.
- Replace ignored options and no-op compatibility stubs with explicit errors.
- Keep a focused test, evaluation, example, and documentation suite for the
  retained behavior.

## Non-goals

- Multi-node inference.
- Ray or externally launched execution.
- CPU, ROCm, XPU, TPU, Ampere, or Blackwell support.
- Models other than DeepSeek V2/V3 and their MTP/EAGLE draft models.
- Checkpoint formats other than Hugging Face safetensors.
- Multimodal, pooling, embedding, reward, classification, speech, or training
  workloads.
- LoRA, structured output, tool calling, generic reasoning parsers, or model
  plugins.
- KV transfer, KV offload, weight transfer, sleep mode, or remote storage.
- Production high availability, worker recovery, elastic scaling, or expert
  load rebalancing.
- Compatibility with removed vLLM configuration fields or public extension
  interfaces.

## Supported contract

### Hardware

- Linux on one machine.
- NVIDIA Hopper GPUs with compute capability SM90.
- All configured ranks must fit on locally visible GPUs.
- NCCL is the GPU collective backend. Gloo may be used only for required local
  CPU coordination.

Startup fails before workers are spawned when these constraints are not met.

### Models and weights

- `DeepseekV2ForCausalLM` and `DeepseekV3ForCausalLM`.
- `DeepSeekMTPModel`, `EagleDeepSeekMTPModel`, and the DeepSeek EAGLE3
  architectures.
- Hugging Face model configuration and tokenizer assets.
- Hugging Face safetensors weights loaded from either a local
  snapshot-compatible directory or a Hugging Face Hub repository ID.
- BF16 and pre-quantized FP8 execution.

PyTorch checkpoints, GGUF, BitsAndBytes, tensorizer, RunAI, dummy loaders,
sharded-state loaders, and loader plugins are not supported.

### Online API

The server exposes only:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health`
- `GET /metrics`

Chat and completion requests support streaming and non-streaming responses.
Generation controls include temperature, top-p, top-k, maximum tokens, stop
conditions, seed, and log probabilities. Unsupported OpenAI fields receive a
request-validation error rather than being ignored.

Responses, MCP, pooling, tokenize/detokenize, beam/scoring, profile, fault
tolerance, SageMaker, and development routes are removed.

### Offline API

The offline surface retains:

- `LLM(...)`
- `LLM.generate(...)`
- `LLM.chat(...)`

Offline inference uses the same V1 engine, scheduler, executor, worker, model
runner, and output processor as online inference. Pooling, scoring, beam search,
and other offline mixins are removed.

## Runtime architecture

```text
Online:  vllm serve -> minimal FastAPI app -> AsyncLLM
                                                \
                                                 -> V1 engine and scheduler
                                                /
Offline: LLM.generate/chat -> synchronous client
                              |
                              v
                    local parallel executor
                              |
                              v
                locally spawned Hopper GPU ranks
                   |       |       |       |
                  TP      PP      PCP     local DP
                   \       |       /       /
                    static EP and DCP groups
                              |
                              v
              DeepSeek MLA/MoE model runner
                              |
                              v
                 base, MTP, or EAGLE decode
                              |
                              v
                    sampler and detokenizer
```

Online and offline requests converge on one internal request representation.
The tokenizer and request processor create engine requests. The scheduler
performs continuous batching and KV-cache allocation. The local executor sends
scheduler outputs to GPU ranks. The model runner performs optimized DeepSeek
execution and speculative verification. Sampled token IDs return through the
engine to a shared output processor. The online path streams formatted OpenAI
events; the offline path collects final Python outputs.

Only the HTTP process formats protocol responses. Only rank zero returns sampled
results from a model-parallel group.

## Parallel execution

The runtime preserves these logical modes:

- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Local data parallelism (DP)
- Static expert parallelism (EP)
- Prefill-context parallelism (PCP)
- Decode-context parallelism (DCP)

The local executor uses `spawn` and creates all ranks on one host. The principal
GPU count is `TP * PP * PCP * DP`. DCP subdivides compatible TP ranks rather
than multiplying the rank count. EP changes MoE expert placement across the
existing ranks.

The implementation retains one mechanism for each concern:

- One local multiprocessing executor for rank lifecycle and RPC.
- One local DP request router and supervisor.
- NCCL device process groups and the minimum required Gloo groups.
- One all-gather/reduce-scatter implementation for expert all-to-all.
- Static linear expert placement.

Ray, external launchers, multi-node rendezvous, remote DP load balancing,
elastic EP, expert load balancing, NIXL, MoRI, DeepEP, and alternate FlashInfer
all-to-all transports are removed.

Configuration validation checks topology divisibility, supported combinations,
visible GPU count, and local port availability before model loading. Invalid
topologies fail with a message that identifies the conflicting sizes.

## Optimized Hopper execution

SM90 is a runtime and build invariant, not a dispatch option. The retained
kernel stack contains only the paths required for DeepSeek BF16/FP8 inference
on Hopper:

- DeepSeek MLA prefill and decode.
- FlashAttention prefill and FlashMLA decode for DeepSeek MLA.
- DeepSeek MoE routing with one Hopper BF16 expert implementation and one
  DeepGEMM FP8 expert implementation.
- Required tensor-parallel linear and vocabulary operations.
- Paged KV-cache operations and prefix-cache reuse.
- MTP/EAGLE drafting and verification.
- `torch.compile` passes used by the retained model.
- CUDA graph capture, replay, and shape management.
- Required kernel warmup and autotuning inputs.

Architecture selectors are replaced by direct Hopper construction. Portable,
Ampere, Blackwell, non-DeepSeek, and fallback implementations are removed.
Compilation or graph-capture failure is reported; execution does not silently
switch to an unrequested backend. An explicit eager configuration remains for
correctness comparison and diagnosis.

## Configuration

The public configuration is reduced to six groups:

1. Model: checkpoint source, tokenizer, BF16/FP8 dtype, maximum model length.
2. Parallel: TP, PP, DP, PCP, DCP sizes and static EP enablement.
3. Cache and scheduling: GPU memory utilization, block size, maximum sequences,
   maximum scheduled tokens, and prefix caching.
4. Optimization: eager/compiled mode, CUDA graph sizes, and warmup controls.
5. Speculation: disabled/MTP/EAGLE mode, draft-token count, and optional EAGLE
   checkpoint.
6. Serving: host, port, API key, request logging, and metrics.

Generic registries, dynamically supplied worker classes, backend names, plugin
hooks, and fields for unsupported features are removed. Unknown CLI flags and
configuration keys are errors.

## Component boundary

### Retained and specialized

- Minimal serving CLI and FastAPI application.
- Offline `LLM` generation and chat.
- Shared tokenizer, request conversion, detokenization, and sampling.
- V1 engine, continuous-batching scheduler, and paged KV-cache manager.
- Local multiprocessing execution and supported parallel groups.
- DeepSeek V2/V3, MTP, EAGLE, and EAGLE3 models.
- Safetensors loading.
- Hopper MLA, MoE, FP8, compilation, CUDA graph, and warmup paths.
- Focused logging and Prometheus metrics.

### Removed

- Other models, tasks, devices, kernels, quantization methods, and loaders.
- Generic model, attention, quantization, executor, and plugin registries.
- Multimodal, LoRA, pooling, structured-output, and tool/reasoning subsystems.
- Generic speculative proposers.
- Alternate serving and offline APIs.
- KV/weight transfer, offload, sleep, profiling, tracing integrations, usage
  telemetry, and fault-tolerance orchestration.
- Compatibility stubs and empty packages left by earlier pruning.
- Unrelated tests, examples, benchmarks, generated documentation, dependencies,
  and build inputs.

Every retained module must be reachable from an approved online/offline root,
be loaded by a retained direct qualname, or support the focused test and build
tooling. Removed packages do not survive solely to keep old imports working.

## Error handling and lifecycle

- Validate hardware, model architecture, weight format, topology, and supported
  options before spawning workers.
- Treat any rank initialization or execution failure as a group failure.
- Terminate all local workers and return a nonzero server process status after a
  worker failure. Automatic recovery is not provided.
- Return OpenAI-style validation errors for invalid online requests.
- Raise typed Python exceptions for invalid offline requests and engine errors.
- Log the model, dtype, parallel topology, optimization mode, and speculation
  mode at startup.
- Never ignore unsupported options or silently substitute a loader, kernel,
  executor, or no-op implementation.

## Repository and learning material

The focused repository keeps:

- An architecture overview and request/token lifecycle.
- A recommended end-to-end source reading order.
- One offline generation example.
- One minimal server example.
- A parallel topology guide.
- A Hopper kernel map.
- Exact environment, test, evaluation, and benchmark commands.
- Required license, security, vulnerability-management, contribution, and agent
  policy documents.

Unrelated upstream feature documentation, examples, benchmarks, and generated
reference material are removed. Documentation must describe only behavior that
exists in the focused tree.

## Reduction sequence

Each stage must leave the approved path runnable:

1. Restore and specialize local multiprocessing execution.
2. Establish online and offline characterization tests.
3. Collapse loading to Hugging Face safetensors.
4. Replace registries and factories with explicit DeepSeek/Hopper wiring.
5. Reduce configuration and CLI arguments.
6. Specialize kernels and dispatch for SM90.
7. Reduce online and offline entrypoints.
8. Simplify every parallel mode to its local implementation.
9. Remove unsupported subsystems and compatibility stubs.
10. Prune tests, documentation, examples, dependencies, and build inputs after
    their runtime consumers are removed.

Large mechanical deletion commits follow the substantive change that makes the
deleted code unreachable. This keeps review and regression diagnosis tractable.

## Verification

### Unit and CPU-safe tests

- Configuration and topology validation.
- Request conversion and sampling validation.
- Scheduler state transitions and KV-cache allocation.
- Prefix-cache behavior.
- Online protocol validation and offline exception behavior.
- Import-closure and package-boundary checks.

### Hopper tests

- MLA prefill/decode correctness.
- MoE routing and FP8 expert correctness.
- BF16/FP8 linear and collective operations.
- Compiled versus eager correctness.
- CUDA graph capture/replay correctness.

### Multi-GPU tests

- Focused TP, PP, DP, EP, PCP, and DCP configurations.
- Representative compatible combinations rather than a combinatorial matrix.
- Rank failure and coordinated shutdown behavior.

### End-to-end tests and evaluation

- BF16 and FP8 safetensors through `LLM.generate()` and `LLM.chat()`.
- Streaming and non-streaming chat/completion HTTP requests.
- Base, MTP, and EAGLE decoding.
- Deterministic token parity between eager and optimized execution where
  deterministic behavior is supported.
- Output-quality evaluation where exact token parity is not appropriate.
- A focused serving throughput and latency comparison.

Test commands and results, model evaluation results, and performance results are
recorded in the eventual change handoff. Python tooling follows the repository
rule: use `uv` and `.venv/bin/python`, never system Python or bare pip.

Static verification rejects imports from removed packages, unsupported backend
names, compatibility no-op stubs, unresolved qualnames, missing package
boundaries, and whitespace errors.

## Acceptance criteria

- The focused tree builds for SM90 CUDA without references to another device or
  GPU generation.
- A local safetensors DeepSeek V2/V3 checkpoint serves all five retained HTTP
  routes.
- Offline `generate()` and `chat()` use the same V1 runtime as the server.
- BF16 and FP8 work in eager and optimized modes.
- Base, MTP, and EAGLE decoding pass focused correctness/evaluation checks.
- TP, PP, local DP, static EP, PCP, and DCP each pass a multi-GPU test.
- Unsupported models, formats, hardware, flags, and routes fail explicitly.
- No removed subsystem remains as a no-op compatibility stub.
- Focused documentation matches the runnable tree.
- Python runtime code is targeted at approximately 120,000 to 170,000 lines;
  focused Python tests are targeted at approximately 25,000 to 45,000 lines.
  These are design budgets, not reasons to remove necessary correctness code.

## Risks and mitigations

### Deletion hides a live dynamic dependency

Mitigation: establish characterization tests first, scan retained qualname
strings and entry points, and remove code only after its caller is specialized.

### Parallel-mode combinations are under-tested

Mitigation: validate topology algebra in unit tests and run representative
combined topologies on a Hopper host in addition to one-mode tests.

### Hopper specialization reduces diagnostic fallback options

Mitigation: retain explicit eager execution, test it against optimized execution,
and fail rather than silently switching kernels.

### Restoring multiprocessing reintroduces broad upstream code

Mitigation: restore the upstream executor only as a behavioral reference, then
remove multi-node, remote-message-queue, Ray, elastic, and unused RPC branches
under focused tests.

### Removing public compatibility surfaces breaks downstream users

Mitigation: treat this as an intentionally focused fork, document its supported
contract prominently, and reject unsupported configuration at startup.
