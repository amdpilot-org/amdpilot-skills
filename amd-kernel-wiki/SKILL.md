---
name: amd-kernel-wiki
description: Use when the user asks about optimizing AMD MI300X (gfx942/CDNA3) or MI355X (gfx950/CDNA4) GPU kernels — MFMA/FP8/MXFP4, AITER, CK (Composable Kernel), hipBLAS, MIOpen, Triton-on-ROCm, warp specialization on RDNA3/CDNA, FlashAttention-AMD, DeepGEMM-AMD, or wants concrete PR references from ROCm/aiter, CK, MIOpen, SGLang-AMD, vLLM-AMD. Do NOT use for generic HIP Q&A that is not CDNA-specific, host-side framework integration, or distributed systems (DeepEP/EPLB/DualPipe).
argument-hint: "[natural-language-question] | [--tag foo --type kernel] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# AMD Kernel Wiki — CDNA3 (MI300X) & CDNA4 (MI355X) Kernel Optimization

Query a structured, cross-referenced knowledge base of GPU kernel optimization for AMD MI300X (gfx942) and MI355X (gfx950). This is the AMD counterpart of `KernelWiki` (which covers NVIDIA Blackwell/Hopper).

## When To Use This Skill

Trigger this skill when the user asks about:

- **CDNA3/CDNA4 kernel programming** — MFMA (Matrix Fused Multiply-Add), FP8 (E4M3/E5M2), MXFP4 (block-scaled FP4, MI355X only), MXFP8, LDS (Local Data Share), wavefront scheduling
- **AMD kernel libraries** — AITER (AMD Instinct Triton Extension Runtime), CK (Composable Kernel), hipBLAS, hipBLASLt, MIOpen, rocRAND, rocThrust
- **Kernel implementations** — FlashAttention-AMD, DeepGEMM-AMD, FlashMLA-AMD, fused MoE (AITER), grouped GEMM (CK), allreduce fusion (AITER), quantization kernels (FP8/MXFP4)
- **Performance patterns** — low CU utilization, memory-bound, register pressure, compute-bound, tail effects, pipeline stalls, LDS bank conflicts
- **DSLs for AMD** — Triton-on-ROCm, CK (Composable Kernel), hip-level C++, HIP Python (numba)
- **MI300X → MI355X migration** — FP8 → MXFP4, gfx942 → gfx950 dispatch, AITER gfx95 gate
- **PR references** — "how did ROCm/aiter / CK / MIOpen / SGLang-AMD / vLLM-AMD implement X for gfx942/gfx950?"
- **Competition solutions** — AMD parity test issues, sglang AMD CI

Do NOT use this skill for:

- Generic HIP questions unrelated to CDNA tensor cores
- Host-side framework integration (model loading, request routing, scheduling policy)
- Distributed systems topics — DeepEP, EPLB, DualPipe are out of scope
- NVIDIA-specific kernel questions (use `KernelWiki` instead)

## gfx942 (MI300X) vs gfx950 (MI355X) at a glance

| Feature | MI300X (gfx942, CDNA3) | MI355X (gfx950, CDNA4) |
|---|---|---|
| Compute Units (CUs) | 304 | 320 |
| HBM | 8 stacks HBM3, 192GB, 5.3 TB/s | 8 stacks HBM3e, 256GB, 8 TB/s |
| FP8 (E4M3/E5M2) MFMA | Yes | Yes |
| MXFP4 (block-scaled FP4) | **No** | Yes |
| MXFP8 (block-scaled FP8) | **No** | Yes |
| BF16 peak (vector) | 163 TF/s | 197 TF/s |
| FP8 peak (matrix) | 1307 TF/s | 3154 TF/s |
| AITER fused MoE (MXFP4) | **No** (gfx95 gate) | Yes |
| AITER allreduce fusion | Yes (no gfx95 gate) | Yes |
| CK (Composable Kernel) | Yes | Yes |
| MIOpen | Yes | Yes |

**Key gotcha:** `_is_gfx95_supported` (in `sglang.srt.layers.communicator`) gates the MXFP4 / fused-MoE AITER paths. MI300X returns `False` and falls back to unfused paths. The AITER allreduce fusion is NOT gfx95-gated.

## How To Query

This skill is a knowledge base (not a script-driven wiki like `KernelWiki`). Query by reading the sections below, or by grepping the referenced source repos.

### Path 1: Read the knowledge sections below

The wiki is organized into sections:
- **AMD kernel libraries** — AITER, CK, hipBLAS, MIOpen
- **Optimization patterns** — common bottlenecks and fixes
- **PR references** — where to find implementations

### Path 2: Grep the source repos

```bash
# AITER (AMD Instinct Triton Extension Runtime)
grep -rn "aiter" /home/xiasun/sglang-fork/python/sglang/srt/layers/

# CK (Composable Kernel)
grep -rn "composable_kernel\|ck_" /home/xiasun/sglang-fork/python/sglang/srt/layers/

# MIOpen (conv / VAE)
grep -rn "miopen\|channels_last_3d" /home/xiasun/sglang-fork/python/sglang/multimodal_gen/
```

### Path 3: Search upstream PRs

```bash
# Search sgl-project/sglang for AMD-related PRs
gh pr list --repo sgl-project/sglang --search "AMD OR ROCm OR AITER OR MI300 OR MI355" --limit 20

# Search ROCm/aiter for kernel implementations
gh pr list --repo ROCm/aiter --limit 20
```

---

## AMD Kernel Libraries

### AITER (AMD Instinct Triton Extension Runtime)
- **Repo:** https://github.com/ROCm/aiter
- **What it is:** AMD's collection of high-performance Triton-on-ROCm kernels for LLM serving — fused MoE, allreduce fusion, RMSNorm, quantization, attention.
- **Key paths in SGLang:**
  - `python/sglang/srt/layers/communicator.py` — `apply_aiter_all_reduce_fusion` (gated on `_use_aiter = SGLANG_USE_AITER && is_hip()`, NOT on gfx95)
  - `python/sglang/srt/layers/moe/moe_runner/aiter.py` — AITER fused MoE runner (gated on `_is_gfx95_supported` for MXFP4)
  - `python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py` — MXFP4 quantization (gfx950 only)
- **gfx942 vs gfx950:** AITER's fused MoE + MXFP4 paths require gfx950 (MI355X). The allreduce fusion + FP8 paths work on gfx942 (MI300X) too.
- **Env var:** `SGLANG_USE_AITER=1` to enable.

### CK (Composable Kernel)
- **Repo:** https://github.com/ROCm/composable_kernel
- **What it is:** AMD's C++ template library for high-performance GEMM/conv/normalization kernels on RDNA/CDNA. The AMD equivalent of CUTLASS.
- **Key use cases in SGLang:** grouped GEMM (for MoE), fused GEMM+activation, conv (via MIOpen which uses CK internally).
- **Porting from CUTLASS:** CK's `DeviceGemm*` instances are the analog of CUTLASS `Gemm*` kernels. The tiling/pipeline concepts map 1:1, but the API is different — CK uses `tensor_descriptor` for global→LDS loads where CUTLASS uses `cute::Tensor`.

### hipBLAS / hipBLASLt
- **Repo:** https://github.com/ROCm/hipBLAS, https://github.com/ROCm/hipBLASLt
- **What it is:** AMD's BLAS libraries. hipBLASLt is the tuned-kernel library (analog of cuBLASLt).
- **Key use cases in SGLang:** linear layers (`Linear`), GEMV, batched GEMM. SGLang's `Linear` layer dispatches to hipBLASLt on ROCm when available.

### MIOpen
- **Repo:** https://github.com/ROCm/MIOpen
- **What it is:** AMD's DNN library (conv, pooling, normalization). The AMD analog of cuDNN.
- **Key use cases in SGLang:** Conv3d for diffusion VAE (LTX-2, FLUX, Wan), `channels_last_3d` (NDHWC) layout.
- **Gotcha:** MIOpen may fall back to NCDHW internally even when `channels_last_3d` is requested. Check with `MIOpen_ENABLE_LOGGING=1`. See issue #50.
- **Auto-tune:** `cudnn.benchmark = True` in SGLang maps to MIOpen auto-tune on ROCm (`rocm.py:optimize_vae`).

---

## Optimization Patterns

### Pattern: Low CU utilization (< 50%)
- **Signal:** `GRBM_GUI_ACTIVE` low, `SQ_WAVES` low
- **Cause:** not enough workgroups to fill 304/320 CUs
- **Fix:** increase workgroup count (smaller tiles), or fuse kernels to amortize launch overhead

### Pattern: HBM-bandwidth-bound
- **Signal:** `TCC_*` counters show high traffic, `SQ_WAIT_LGVM` dominant, low L1 hit rate
- **Cause:** memory access not coalesced, or working set too large for L2
- **Fix:** coalesce wavefront-wide loads, use LDS staging, tile for L2 reuse

### Pattern: LDS-latency-bound
- **Signal:** `SQ_WAIT_LGVM` dominant, but L1 hit rate high, HBM traffic low
- **Cause:** LDS (shared memory) load latency
- **Fix:** double-buffer LDS loads, prefetch with `buffer_load`, overlap compute with LDS traffic

### Pattern: Register pressure
- **Signal:** `SQ_WAIT_SGPR` dominant, low occupancy
- **Cause:** too many live registers per wavefront
- **Fix:** spill to LDS, reduce loop-carried deps, reduce workgroup size (more waves per CU)

### Pattern: Not using MFMA (tensor cores)
- **Signal:** low `SQ_INSTS_VALU_MFMA`, high `SQ_INSTS_VALU_*` (vector ALU)
- **Cause:** kernel not routed through hipBLAS/AITER/CK
- **Fix:** check `SGLANG_USE_AITER=1`, check gfx942 vs gfx950 gate, route through `aiter` / `CK` / `hipBLASLt`

### Pattern: FP8 path not dispatched (MI300X)
- **Signal:** low FP8 MFMA, high BF16 MFMA
- **Cause:** `quantization=fp8` not set, or kernel doesn't support gfx942 FP8
- **Fix:** verify `quantization=fp8` in server args, check kernel's gfx942 FP8 support

### Pattern: MXFP4 path not dispatched (MI355X)
- **Signal:** low MXFP4 MFMA, high FP8 or BF16 MFMA
- **Cause:** `_is_gfx95_supported` returns False, or `quantization=mxfp4` not set
- **Fix:** verify gfx950 (`rocminfo | grep Name`), check `SGLANG_USE_AITER=1`, verify `quantization=mxfp4`

### Pattern: Tail effect (utilization drops at end)
- **Signal:** `rocprof --trace` shows utilization drop in last 10% of kernel
- **Cause:** load-imbalance on variable-length inputs (common in attention with variable seq lengths)
- **Fix:** pad inputs, or split into balanced workgroups

### Pattern: numa_balancing=1 hang (MI300X)
- **Signal:** kernel hangs (not crashes) during hicache alloc, TP-wide deadlock
- **Cause:** `numa_balancing=1` (default on some kernels) causes file-backed pages to bypass MPOL_BIND
- **Fix:** `echo 0 > /proc/sys/kernel/numa_balancing` (the ONLY complete fix). See [[project_sglang_numa_binding_amd_gap]].

---

## PR References

### SGLang AMD-related PRs
- Search: `gh pr list --repo sgl-project/sglang --search "AMD OR ROCm OR AITER" --limit 30`
- Key files: `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/moe/moe_runner/aiter.py`, `python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py`, `python/sglang/srt/hardware_backend/rocm.py`

### ROCm/aiter PRs
- Search: `gh pr list --repo ROCm/aiter --limit 30`
- Key kernels: `allreduce_fusion_kernel_1stage`, `fused_moe`, `rmsnorm_quant`

### CK (Composable Kernel) PRs
- Search: `gh pr list --repo ROCm/composable_kernel --limit 30`
- Key instances: `DeviceGemmXdl`, `DeviceConvFwdXdl`

### MIOpen PRs
- Search: `gh pr list --repo ROCm/MIOpen --limit 30`
- Key: `Conv3d` NDHWC kernels

---

## Related skills

- [`KernelWiki`](https://github.com/mit-han-lab/KernelWiki) — NVIDIA counterpart (Blackwell/Hopper kernel optimization)
- [`rocprof-report-skill`](../rocprof-report-skill/) — AMD kernel profiling (rocprof/omniperf)
- [`debug-gpu-crash`](../debug-gpu-crash/) — AMD crash debugging
- [`llm-torch-profiler-analysis`](../llm-torch-profiler-analysis/) — higher-level torch.profiler triage (vendor-agnostic)

---

## References

- MI300X architecture: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- MI355X architecture: https://www.amd.com/en/products/accelerators/instinct/mi355/mi355x.html
- ROCm docs: https://rocm.docs.amd.com/
- AITER docs: https://github.com/ROCm/aiter#readme
- CK docs: https://github.com/ROCm/composable_kernel#readme
- MIOpen docs: https://github.com/ROCm/MIOpen#readme
- CDNA3 whitepaper: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html#tech-docs
- CDNA4 whitepaper: https://www.amd.com/en/products/accelerators/instinct/mi355/mi355x.html#tech-docs
