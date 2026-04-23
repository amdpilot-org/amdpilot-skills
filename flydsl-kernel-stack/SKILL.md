---
name: flydsl-kernel-stack
description: >
  Meta-index for the FlyDSL fused-MoE stack on AMD ROCm (MI300X/MI350/MI355X).
  FlyDSL is a Python DSL backed by a custom MLIR pipeline that emits optimized
  binaries for gfx942/gfx950. Use this skill whenever the task mentions FlyDSL,
  DSL2_ROOT, AITER_USE_FLYDSL_MOE, FLYDSL_W4A16_HYBRID, fly dialect, flyc.kernel,
  kimi-k2.5 moe, or the AITER `dev/kimi-K2.5` branch. Routes into the specialist
  skills for build, debug, kernel authoring, tile programming, gemm, and lds.
---

# FlyDSL Kernel Stack (Meta-Index)

**Scope:** how to *use* FlyDSL to accelerate AMD kernels (especially the Kimi-K2.5
fused-MoE path), not how to be FlyDSL. Upstream docs at
[ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) remain authoritative.

## 1. What the stack actually is

```
Python (@flyc.kernel, layout algebra)
  └─> MLIR fly dialect
        └─> ROCDL
              └─> fatbin (gfx942 / gfx950)
```

FlyDSL is a Python front-end (PyPI package `flydsl`) backed by a custom MLIR
stack. FLIR (Flexible Layout IR) is a `(Shape, Stride)` layout algebra
inspired by CuTe that makes tiling / swizzling / vectorization composable.

Compilation pipeline: canonicalization → CSE → GPU-to-ROCDL lowering → fatbin.
Targets supported by upstream `main`: `gfx942` (MI300X) + `gfx950` (MI350/MI355X).

## 2. Container paths (immutable in the K2.6 eval)

| Path | Env var | Purpose |
|---|---|---|
| `/opt/FlyDSL` | `DSL2_ROOT` | Where AITER adds FlyDSL to `sys.path` |
| `/opt/mlir_install` | `MLIR_PATH` | Prebuilt MLIR the FlyDSL compiler links against |
| `/sgl-workspace/aiter` | — | AITER editable install, must be on `dev/kimi-K2.5` |
| `/sgl-workspace/sglang` | — | SGLang editable install |
| `~/.flydsl/cache/` | `FLYDSL_RUNTIME_ENABLE_CACHE` | JIT kernel cache (per-user) |

## 3. Gating env vars (master table)

These are **gates**, not tunables. Every scored optimization trial must also
include a tracked source edit in AITER or SGLang — env-var-only toggling is
rejected by the supervisor.

| Variable | Value | Purpose |
|---|---|---|
| `AITER_USE_FLYDSL_MOE` | `1` enable / `0` baseline | Master gate for the FlyDSL MoE path |
| `AITER_USE_FLYDSL_MOE_STAGE1` | `1` | Replace gate/up projection (Stage 1) |
| `AITER_USE_FLYDSL_MOE_STAGE2` | `1` | Replace down projection (Stage 2) |
| `AITER_ENFORCE_DSL` | `1` | Hard-fail on FlyDSL import error (avoids silent fallback) |
| `FLYDSL_W4A16_HYBRID` | unset / `w2_bf16` | `w2_bf16`: Stage1 W4A16, Stage2 BF16 |
| `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT` | `3` | Blog-recommended CK tile conversion mode |
| `TRITON_MAX_CACHE_SIZE` | `2147483648` | 2 GiB Triton cache ceiling |
| `AITER_FLYDSL_MOE_COMPARE` | `0` | Disable per-call correctness compare (debug only) |
| `AITER_FLYDSL_MOE_COMPARE_STAGE2` | `0` | Same, Stage 2 |
| `AITER_FLYDSL_DEBUG` | `0` | Silence debug logging |

## 4. Phased optimization plan (Kimi-K2.6 on MI355X)

Each phase is one or more scored trials. Only proceed to the next phase after
the current phase's best config is measured and recorded.

| Phase | Config | Source edit expected |
|---|---|---|
| A | verify import of `kernels.moe_gemm_2stage` on gfx950 | AITER branch pin / sys.path fix |
| B | `AITER_USE_FLYDSL_MOE_STAGE1=1` only | Hook into AITER fused_moe dispatcher |
| C | Stage 1 + Stage 2 | Stage-2 down-proj hook |
| D | sweep `FLYDSL_W4A16_HYBRID` ∈ {unset, `w2_bf16`} | Dispatcher branch per hybrid mode |
| E | add `--enable-torch-compile` | server-flag wiring |
| F | add `--disable-radix-cache` (random-input only) | server-flag wiring |
| G | GSM8K `exact_match_flexible >= 0.90` | none — accuracy gate |

**Guard:** BS=1 decode `output_throughput` at input=8192 must stay within
**2%** of PR [sgl-project/sglang#23381](https://github.com/sgl-project/sglang/pull/23381)
baseline (~38.05 tok/s). Any FlyDSL config that wins at concurrency=40 but
regresses BS=1 decode by >2% is rejected.

## 5. Skill router

| Symptom | Skill |
|---|---|
| "How do I build FlyDSL / where is MLIR / PYTHONPATH fails" | [flydsl-build](../flydsl-build/SKILL.md) |
| "Kernel compiles but fails at runtime / gfx target mismatch / silent fallback" | [flydsl-debug-kernel](../flydsl-debug-kernel/SKILL.md) |
| "I need to write a new `@flyc.kernel`" | [flydsl-kernel-authoring](../flydsl-kernel-authoring/SKILL.md) |
| "Layout algebra / tiled copy / partition confusion" | [flydsl-tile-programming](../flydsl-tile-programming/SKILL.md) |
| "GEMM-specific / MFMA / preshuffle / MoE 2-stage" | [flydsl-gemm-optimization](../flydsl-gemm-optimization/SKILL.md) |
| "LDS bank conflicts / swizzling / smem_allocator" | [flydsl-lds-optimization](../flydsl-lds-optimization/SKILL.md) |

## 6. Blog reference numbers (MI300X, K2.5 — directional only)

FlyDSL vs Triton for `fused_moe` kernel in isolation on the Kimi-K2.5 "large"
shape `(tokens=16384, model_dim=7168, inter_dim=512, E=384, topk=8)`:

| dtype | Triton (ms) | FlyDSL (ms) | Speedup |
|---|---|---|---|
| BF16 (A16W16) | 12.09 | 8.68 | 1.39x |
| W4A16 | 31.43 | 9.77 | 3.22x |

End-to-end on MI300X at concurrency=40 (Kimi-K2.5, TP=8):
- Output throughput: 135.39 → 355.35 tok/s (**+162.4%**)
- TPOT mean: 230.37 → 70.86 ms/token (**-69.2%**)
- TTFT mean: 33478.68 → 17730.03 ms (**-47.0%**)
- GSM8K accuracy: 0.96 → 0.96 (unchanged)

MI355X numbers for K2.6 are NOT yet published; Phase 1 of the eval measures
them once and pins `baseline_contract.expected_metric`. **Target on MI355X is
+30% at concurrency=40 (conservative vs blog's +162% on MI300X)**.

## 7. When FlyDSL is NOT the right tool

- Pure elementwise fusions → use Triton (lower compile overhead).
- Workloads where `fused_moe` is <20% of GPU time → profile first, then pick the actual hotspot.
- Adding a new MMA atom for a new arch → compile cycle is 30+ min, out-of-scope for a one-shot optimize run.
- Kernels that need to interop with CUDA graphs that FlyDSL JIT recompiles on first call → warm up before capture.

## 8. Key upstream references

- Repo: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)
- AITER integration: [ROCm/aiter `dev/kimi-K2.5`](https://github.com/ROCm/aiter/tree/dev/kimi-K2.5)
- SGLang baseline PR (K2.6 on MI355X): [sgl-project/sglang#23381](https://github.com/sgl-project/sglang/pull/23381)
- Blog: [Accelerating Kimi-K2.5 on MI300X](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)
- Blog docker tag (MI300X reference only): `clementlincf/amdafde:v0.5.8-rocm720-mi30x-kimi-k2.5-opt-20260224`
