---
name: amd-kernel-optimization
description: >
  Optimize inference latency and throughput of PyTorch models on AMD GPUs (MI250/MI300/MI350) with ROCm.
  Use when profiling and optimizing GEMM, attention, elementwise ops, torch.compile, CUDAGraphs,
  or Triton kernels on AMD hardware. Covers the full optimize cycle: benchmark → profile → analyze →
  implement → verify. Also covers benchmarking methodology and common pitfalls that waste time.
---

# AMD Kernel Optimization (ROCm)

## 3 Rules (read first)

1. **Establish a baseline, then make `torch.compile` the first major optimization.** `torch.compile(mode="default")` with correct inductor config gives 2-5x speedup. Get a reproducible baseline first, then make compile work before any manual kernel surgery. Any code change that breaks compile is a net regression.

2. **Profile before manual optimization.** Never guess where time is spent. If CUDA graphs hide kernels, create a fast profiling path with graphs disabled, run `torch.profiler`, classify GPU time into GEMM / attention / elementwise / launch overhead, then optimize the largest category.

3. **Measure after every change.** Benchmark with proper warmup and iterations (see below). Revert if performance regresses.

## Benchmarking (do this correctly or waste hours)

**NEVER reduce warmup/iterations to "save time" — you get garbage numbers.**

- **Minimum**: 3 warmup runs, 10 measurement iterations. Use the benchmark script's defaults.
- **Use GPU timing**, not wall-clock:
  ```python
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record(); result = model(input); end.record()
  torch.cuda.synchronize()
  ms = start.elapsed_time(end)
  ```
- **Report mean AND std.** If std > 10% of mean, something is wrong (graph breaks, recompilation).
- **First-run penalty is NORMAL on AMD** — torch.compile takes 2-15 min on first run. Set timeout ≥ 600s. Do NOT conclude "compile doesn't work" or kill jobs under 15 min.
- **Never report first-run latency as baseline.** Always use mean of post-warmup iterations.

## Source-Level Changes Are Required

**Env-var-only or config-flag-only optimization does NOT count as a valid optimization attempt.**
Once a baseline is established and profiling identifies the hotspot, every scored optimization
trial must include at least one tracked source code change. This means editing actual Python,
C++, Triton, or HIP source files — not just setting environment variables or CLI flags.

Examples of valid source-level changes:
- Editing GEMM backend dispatch logic to select a faster kernel for the workload's shapes
- Writing or tuning a Triton kernel for an elementwise fusion
- Modifying attention implementation to use a faster backend
- Changing CK/hipBLASLt tile configs or kernel instance factories
- Fusing model operations (QKV projection fusion, gate+up fusion)

Examples of changes that are NOT sufficient on their own:
- Setting `PYTORCH_TUNABLEOP_ENABLED=1` or `TORCH_BLAS_PREFER_HIPBLASLT=1`
- Passing `--use-flash-attn` as a CLI flag
- Changing batch size or sequence length parameters
- Setting env vars without touching source code

## Optimization Ladder

Each level builds on the previous. **When the baseline is already established and the
environment is pre-configured (most optimization tasks), skip Level 1 and start from
profiling → Level 3+.** Only fall back to Level 1 if env vars are clearly misconfigured.

### Level 1: Environment (only if not already configured)
- Set env vars: `GPU_MAX_HW_QUEUES=2`, `HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `AMD_LOG_LEVEL=0`
- Disable NUMA balancing: `sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'`
- Enable GEMM tuning: `PYTORCH_TUNABLEOP_ENABLED=1`, `TORCH_BLAS_PREFER_HIPBLASLT=1`
- Set `torch.set_float32_matmul_precision('high')`
- Audit env vars: `env | grep -iE 'TORCH|INDUCTOR|AUTOTUNE'` — unset `TORCHINDUCTOR_MAX_AUTOTUNE` if present
- **Skip this level** if the Docker image or benchmark script already sets these. Check first.

### Level 2: torch.compile (skip if already enabled in the benchmark)
- Apply inductor config, compile with `mode="default"`. Details: [references/torch-compile-and-graphs.md](references/torch-compile-and-graphs.md)
- Fix ALL graph breaks before Level 3: `TORCH_LOGS="graph_breaks" python3 ...`
- If kernel launch overhead is high after compile, try manual CUDAGraph capture (see reference)
- Before Level 3, capture a profiling snapshot so you know what compile fixed and what still dominates
- This alone typically gives 2-5x speedup
- **Skip this level** if the benchmark already uses torch.compile and the baseline is solid

### Level 3: Model surgery and backend switching (start here when baseline is established)
- **Profile first** → read [references/benchmarking-and-profiling.md](references/benchmarking-and-profiling.md)
- **Backend switching is a first-class optimization** — edit the dispatch/selection code to route
  through a faster backend (CK, hipBLASLt, Triton, aiter) for the workload's dominant shapes.
  This is a *source-level change* to the backend selection logic, not an env var.
- Fuse QKV projections (3 GEMMs → 1), Gate+Up projections (2 GEMMs → 1) → [references/gemm-and-linear.md](references/gemm-and-linear.md)
- Replace manual attention with SDPA or aiter flash attention → [references/gemm-and-linear.md](references/gemm-and-linear.md)
- Look for compute-reduction: skip masked/padding inputs, avoid `repeat_kv`, cache unchanged outputs
- **Test**: `TORCH_LOGS="graph_breaks"` — verify no new breaks after each change

### Level 4: Kernel-level optimization (Triton, CK tile configs, custom kernels)
- Write Triton kernels for elementwise fusions (RMSNorm, SiLU+Mul, Add+RMSNorm) → [references/triton-on-rocm.md](references/triton-on-rocm.md)
- Route fused GEMMs through aiter tuned GEMM with M-threshold gating → [references/gemm-and-linear.md](references/gemm-and-linear.md)
- **Edit CK tile configs** — modify kernel config headers (`ck_grouped_gemm_kernel_config.h`) or
  instance factory files to add gfx950-specific tile sizes tuned for the workload's GEMM shapes
- Inductor tuning flags: `coordinate_descent_tuning`, `benchmark_kernel`, `freezing` (increase compile time, improve steady-state)

### Level 5: Architecture-specific kernels
- aiter kernels for attention/GEMM — use `torch.ops.aiter.*` (compile-safe) not Python wrappers
- Weight preshuffling for asm paths (benchmark: may help or hurt per shape)
- If custom kernel breaks compile, wrap with `@torch.compiler.disable` as last resort

## Grouped GEMM / MoE Expert Optimization

For MoE workloads where grouped GEMM dominates GPU time (common in large MoE models), the
optimization target is the grouped GEMM dispatch and kernel selection logic, not just standalone
GEMM backends.

**Key source files to examine:**
- The backend dispatch/selector that decides CK vs hipBLASLt vs Triton for grouped GEMM
- The grouped-MLP implementation (e.g. `PrimusTurboGroupedMLP` or equivalent)
- CK kernel configs and instance factories for the target GPU architecture
- Permutation/unpermute kernels for MoE token dispatch (often 15-20% of total GPU time)

**Optimization approach:**
1. Profile to confirm grouped GEMM dominates (typically 20-40% of total GPU time)
2. Identify which backend is currently active (CK, hipBLASLt, Triton)
3. Edit the backend selection code to try alternatives — this is a *source code change*
4. For CK backends: edit tile config headers to add architecture-specific tile sizes
5. For permutation/dispatch hotspots: look for fusion opportunities in the permutation kernels

**Constraint awareness:**
- Some grouped GEMM paths are coupled with sync-free MoE stages. Changing the GEMM backend may
  require coordinated changes to the sync-free stage or DeepEP token-count contracts.
- Check for hard validation checks that reject incompatible flag combinations before making changes.

## AMD-Specific Alternatives Quick Reference

### GEMM / Linear
| Option | Notes |
|---|---|
| rocBLAS (default) | Vendor BLAS; generally well-tuned |
| hipBLASLt | Fused epilogues; may beat rocBLAS for some shapes. Note: some frameworks register hipBLASLt with `autotune=False` — it won't win autotune unless explicitly selected or patched. |
| aiter tuned GEMM | Auto-dispatches best kernel per (M,N,K) from tuned configs |
| CK GEMM | AMD Composable Kernel — tile configs in header files are architecture-specific |
| FP8 GEMM (MI300+) | `gemm_a8w8` via aiter; gfx942=`e4m3fnuz`, gfx950=`e4m3fn` |
| Triton FP8 GEMM | `--fp8-gemm-backend triton` — may be faster for decode (small M) on gfx950 |

**Backend switching is a source-level optimization**, not config tuning. Edit the dispatch
code that selects which backend runs for each GEMM shape. Decision tree:
1. Profile first → identify top hotspot kernel and its backend
2. If hotspot is CK GEMM and workload is decode (small M) → try Triton FP8 GEMM
3. If hotspot is "Other/Triton" → check Triton config JSON for gfx950 entries
4. Do NOT spend >3 rounds tuning configs within one backend without profiling verification

**WARNING — FP8 dtype differs by GPU architecture:**
- gfx942 (MI300X): `torch.float8_e4m3fnuz` — hardware only supports FNUZ
- gfx950 (MI355X): `torch.float8_e4m3fn` — hardware supports both, aiter chooses IEEE
- Do NOT assume `is_fp8_fnuz()` should return True for all AMD GPUs
- Check: `aiter/ops/triton/utils/types.py` → `get_fp8_dtypes()`

### Attention
| Option | Notes |
|---|---|
| aiter flash attention | `torch.ops.aiter.mha_fwd.default(...)` — compile-friendly, GQA native |
| SDPA | `F.scaled_dot_product_attention(...)` — good for KV-cache decode |
| Manual bmm+softmax+bmm | Slowest; replace with SDPA |

### Compilation & Graphs
| Option | Notes |
|---|---|
| `torch.compile(mode="default")` | **Start here.** Stable on ROCm with correct inductor config |
| Manual CUDAGraph capture | Wrap full inference in one graph; needs Dynamo RNG patch |
| `reduce-overhead` / `max-autotune` | **Avoid on ROCm** unless you have verified stability |

## Common Pitfalls

- **Relying on env vars / CLI flags instead of source changes**: Setting env vars like `PYTORCH_TUNABLEOP_ENABLED=1` or passing `--use-flash-attn` is NOT optimization. Edit the actual source code — backend dispatch logic, kernel configs, model implementation files. Env-var-only trials with no source edits do not count as optimization attempts.
- **Optimizing without profiling**: Classify kernels by category (GEMM/attention/elementwise/other) and compute percentages. "Top 10 kernels" is not enough.
- **Skipping torch.compile**: Manual fusion that saves 10% is worthless if you're missing the 3x from compile. Get compile working first — but if compile is already enabled in the baseline, skip to source-level optimization.
- **Giving up on first failure**: When a technique causes regression, diagnose and adjust (e.g., M-threshold gating for aiter GEMM), don't abandon it entirely.
- **Treating blockers as dead ends**: "CUDAGraph doesn't support control flow" means "refactor the control flow." "Requires editing HuggingFace modeling file" means "go edit the modeling file" — that IS the work. Source-level changes are the goal, not the obstacle.
- **Not modifying inner model layers**: The hottest code is in attention and MLP modules, often in third-party libraries. Locate them: `python -c "import transformers; print(transformers.__file__)"` and edit directly. Editing library internals is expected and encouraged.
- **Testing only in isolation**: Optimizations compose. A technique showing 0% alone may enable others. Build incrementally — each new technique applied ON TOP of all previous ones.
- **Reducing benchmark parameters**: Setting `WARMUP=0 ITERATIONS=1` gives meaningless numbers. Optimize the code, not the test.
- **aiter tuned GEMM with no tuned configs**: Default config CSV ships empty — `gemm_a16w16` silently falls back to `F.linear` (no error, no benefit). Diagnose: `AITER_LOG_TUNED_CONFIG=1`; if "using torch solution:0", generate configs or use `PYTORCH_TUNABLEOP_ENABLED=1`. Full workflow: [references/gemm-and-linear.md](references/gemm-and-linear.md).
- **Blind backend tuning without decode profiling**: If >40% of decode GPU time is "Other" / unidentified, do NOT proceed to GEMM config tuning. Fix profiling first (see gpu-profiling skill). The Phase 2C GLM-5.1 experiment spent 6 trials tuning CK GEMM configs while 46% of decode time was unprofiled Triton kernels.
- **aiter Triton GEMM configs may be suboptimal for your workload**: aiter Triton kernels use JSON config files at `aiter/ops/triton/configs/gemm/` with M-threshold entries. Default `"any"` fallbacks are typically tuned for large-batch training and may be poor for small-batch inference. Profile to find the dominant GEMM kernels, check their configs, and tune `BLOCK_SIZE_M` and `NUM_KSPLIT` for the actual runtime M value. Details: [references/gemm-and-linear.md](references/gemm-and-linear.md).
- **Treating backend env vars as optimization**: `PRIMUS_TURBO_GROUPED_GEMM_BACKEND` or similar env vars can force backend selection for diagnostics, but the scored optimization must land as a tracked source code edit (e.g., editing the dispatch function, adding an architecture-specific tile config, patching the backend selection logic).

## Reference Files

Read as needed for implementation details:

- **[benchmarking-and-profiling.md](references/benchmarking-and-profiling.md)** — Proper measurement, GPU timing, profile interpretation, what to look for in traces
- **[torch-compile-and-graphs.md](references/torch-compile-and-graphs.md)** — Inductor config, graph breaks and fixes, manual CUDAGraph capture, Dynamo RNG patch, env var audit
- **[gemm-and-linear.md](references/gemm-and-linear.md)** — GEMM backend APIs, aiter tuned GEMM, projection fusion, attention backends, weight preshuffling
- **[triton-on-rocm.md](references/triton-on-rocm.md)** — ROCm Triton gotchas, kernel templates for RMSNorm, SiLU+Mul, GELU+Mul, Add+RMSNorm
