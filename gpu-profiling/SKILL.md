---
name: gpu-profiling
description: >
  GPU kernel profiling workflow for AMD ROCm — profile BEFORE optimizing.
  Covers torch.profiler setup, CUDA graph disabling for kernel visibility,
  trace JSON parsing, and kernel category classification. Use this skill
  whenever you need to identify which GPU kernels dominate execution time.
---

# GPU Profiling on AMD ROCm

**Rule: ALWAYS profile before manual optimization changes.**
Blind optimization without profiling data is the #1 failure mode.
If the task clearly points to `torch.compile` as the first major lever, you can
try compile early, but you still need profiling data before doing manual
kernel/model surgery.

## 1. CUDA Graphs Hide Kernel Details

When CUDA graphs are enabled, the profiler only shows `hipGraphLaunch` —
individual kernel timings are invisible. You MUST profile with CUDA graphs
disabled first, then re-enable for the final benchmark.

Most frameworks support a `--disable-cuda-graph` flag or equivalent env var.

## 2. torch.profiler for Kernel Breakdown

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

## Kernel Categories

| Category | Kernel Pattern | Optimization Target |
|----------|---------------|-------------------|
| **GEMM** | `gemm_`, `_gemm_`, `hipblas`, `rocblas` | Block sizes, data types, tiling |
| **MoE** | `fused_moe`, `fmoe_`, `moe_align` | Expert routing, block_m, ksplit |
| **Attention** | `flash_attn`, `tilelang`, `nsa_`, `mla_` | Backend selection, head grouping |
| **AllReduce** | `allreduce`, `ncclKernel`, `CustomAR` | Fusion, algorithm selection |
| **Elementwise** | `rms_norm`, `silu`, `rope` | Kernel fusion, aiter ops |

## Key Principle

Focus ALL optimization effort on the category with the highest percentage
of GPU time. Optimizing a 5% kernel gives at most 5% improvement;
optimizing a 60% kernel can give up to 60% improvement.
