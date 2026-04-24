---
name: gpu-profiling
description: >
  GPU kernel profiling workflow for AMD ROCm — profile BEFORE optimizing.
  Covers torch.profiler (the primary method), framework-native profiling
  endpoints (sglang / vLLM), CUDA-graph disabling, Triton kernel identification
  via cache inspection, decode vs prefill separation, and structured artifact
  requirements. Use this skill whenever you need to identify which GPU kernels
  dominate execution time.
---

# GPU Profiling on AMD ROCm

**Rule: ALWAYS profile before manual optimization changes.**
Blind optimization without profiling data is the #1 failure mode.
If the task clearly points to `torch.compile` as the first major lever, you can
try compile early, but you still need profiling data before doing manual
kernel/model surgery.

**Default tool: `torch.profiler`.** It captures GPU kernel timings on ROCm
correctly and is sufficient for almost all optimisation decisions on AMD
workloads (training and serving alike). Historical attempts to reach for
`rocprofv3 --attach` / wrap-launch modes on AMD have routinely OOMed on
large models or produced no output under multi-process torchrun. Prefer
`torch.profiler` and only escalate to HSA-level tools when the
kernel-category breakdown genuinely cannot identify the hotspot — and even
then, consider Triton cache inspection and NVTX markers first (see §5 and §7).

## 1. CUDA Graphs Hide Kernel Details

When CUDA graphs are enabled, the profiler only shows `hipGraphLaunch` —
individual kernel timings are invisible. You MUST profile with CUDA graphs
disabled first, then re-enable for the final benchmark.

Most frameworks support a `--disable-cuda-graph` flag or equivalent env var.

## 2. torch.profiler for Kernel Breakdown (primary method)

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Run the target workload here (forward pass, decode steps, etc.)
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("/tmp/trace.json")
```

For iterative workloads (training loops, serving decode), wrap a few
iterations — not the full run. Skipping the first 1-2 iterations avoids
CUDA graph capture and compile time in the profile.

## 3. Parse Trace JSON for Kernel Breakdown

```python
import json

with open("/tmp/trace.json") as f:
    events = json.load(f)["traceEvents"]

gpu_kernels = {}
for e in events:
    if e.get("cat") == "kernel":
        name = e["name"]
        dur = e.get("dur", 0)  # microseconds
        gpu_kernels[name] = gpu_kernels.get(name, 0) + dur

total = sum(gpu_kernels.values())
for name, dur in sorted(gpu_kernels.items(), key=lambda x: -x[1])[:15]:
    pct = dur / total * 100 if total else 0
    print(f"  {pct:5.1f}%  {dur/1000:8.1f}ms  {name}")
```

This gives you the kernel-category breakdown needed to decide where to
optimize next — GEMM vs MoE vs attention vs elementwise. That is the decision
you need; per-kernel HSA-level detail rarely changes the answer.

## 4. Framework-Native Profiling (sglang / vLLM)

For serving frameworks, prefer the built-in profiling endpoint. This runs
`torch.profiler` inside each TP worker process — no ptrace needed, no OOM
risk, produces per-rank traces.

```bash
# sglang: Start profiling (server must be running)
curl -X POST http://localhost:30000/start_profile

# Send your decode requests here...

# Stop profiling — traces saved to /tmp/ in the container
curl -X POST http://localhost:30000/stop_profile
```
Output: per-rank Chrome trace files (`.trace.json.gz`). Parse with the
trace JSON method in §3 above.

**Limitation**: torch.profiler may show Triton kernels as generic
`main_kernel` names. Resolve via the cache inspection method in §5 — do NOT
fall back to rocprofv3 on AMD; it is not reliable enough for large models.

## 5. Triton `main_kernel` Opaque Naming

Triton compiles all kernels to `main_kernel` variants. Two reliable ways to
identify which Triton kernel is which:

1. **Triton cache inspection**: `ls ~/.triton/cache/` →
   `cat ~/.triton/cache/<hash>/metadata.json`. The cache stores the source
   function and tuning key for every compiled Triton kernel.
2. **Autotuning log**:
   `TRITON_PRINT_AUTOTUNING=1 python3 script.py 2>&1 | grep kernel`.
   This logs every autotune decision with the underlying Python function
   name, so you can map `main_kernel` occurrences back to their source.

## 6. Decode vs Prefill: Profile Separately

Prefill and decode have different hotspots:
- **Prefill**: Large M → GEMM-dominated → CK/hipBLAS may be optimal
- **Decode**: Small M (= batch_size) → memory-bound → Triton small-M kernels may be better

```bash
# Prefill-dominated workload
python3 bench.py --num-prompts 1 --input-len 1024 --output-len 1

# Decode-dominated workload
python3 bench.py --num-prompts 1 --input-len 16 --output-len 512
```

For training workloads, the equivalent split is
**forward vs. backward + optimiser step** — wrap each in its own
`torch.profiler` section when you need per-phase attribution. NVTX markers
(`torch.cuda.nvtx.range_push("fwd")` / `range_pop()`) make the trace
timeline much easier to read.

## 7. Profiling Failure = Blocker, Not Footnote

If `torch.profiler` fails or returns opaque data:
- **Do NOT proceed to blind tuning.**
- If the model OOMs under the profiler, reduce the profiled scope: run fewer
  iterations, a smaller microbench, or a single TP rank via
  `HIP_VISIBLE_DEVICES=0`.
- If >40% of GPU time shows as "Other" / unidentified, escalate by inspecting
  the Triton cache (see §5) or by instrumenting the model with
  `torch.cuda.nvtx.range_push` markers so the profiler groups operators by
  logical phase.

## 8. Required Profiling Artifacts

After profiling, you MUST write these to `optimization_state.json`:
- `profiler_command`: exact command used
- `profile_output_path`: path to the produced artifact
- `profiling_scope`: `decode_fullstack` | `prefill_only` | `layer_microbench` | `reduced_memory_decode` | `train_iter` | `fwd_only` | `fwd_bwd`
- `top_kernels`: list of `{"name": str, "pct": float, "backend": str}`
- `hotspot_hypothesis`: what you believe is the bottleneck
- `next_move_grounded_in`: explicit link from hotspot → planned experiment

If profiling failed, record `profiling_failure_mode`: `oom` | `tool_not_found` | `graph_obscured` | `no_kernel_names`

## Kernel Categories

| Category | Kernel Pattern | Optimization Target |
|----------|---------------|-------------------|
| **GEMM** | `gemm_`, `_gemm_`, `hipblas`, `rocblas` | Block sizes, data types, tiling, **backend switching** |
| **MoE** | `fused_moe`, `fmoe_`, `moe_align` | Expert routing, block_m, ksplit |
| **Attention** | `flash_attn`, `tilelang`, `nsa_`, `mla_` | Backend selection, head grouping |
| **AllReduce** | `allreduce`, `ncclKernel`, `CustomAR` | Fusion, algorithm selection |
| **Elementwise** | `rms_norm`, `silu`, `rope` | Kernel fusion, aiter ops |
| **Triton** | `main_kernel` (resolve via cache inspection / TRITON_PRINT_AUTOTUNING=1) | Backend switching, autotuning |

**WARNING — FP8 dtype differs by GPU architecture:**
- gfx942 (MI300X): `torch.float8_e4m3fnuz`
- gfx950 (MI355X): `torch.float8_e4m3fn`
- Do NOT assume all AMD GPUs use the same FP8 format.
- Check: `aiter/ops/triton/utils/types.py` → `get_fp8_dtypes()`

## Key Principle

Focus ALL optimization effort on the category with the highest percentage
of GPU time. Optimizing a 5% kernel gives at most 5% improvement;
optimizing a 60% kernel can give up to 60% improvement.

**Backend switching** (e.g. CK GEMM → Triton FP8 GEMM) is a separate
optimization lever from config tuning within a backend. If the top hotspot
is a GEMM kernel, try switching backends before spending time on config tuning.
