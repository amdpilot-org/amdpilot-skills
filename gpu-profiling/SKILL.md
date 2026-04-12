---
name: gpu-profiling
description: >
  GPU kernel profiling workflow for AMD ROCm — profile BEFORE optimizing.
  Covers torch.profiler, rocprofv3 kernel tracing, CUDA graph disabling,
  Triton kernel identification, decode vs prefill separation, large model
  OOM workarounds, and structured artifact requirements. Use this skill
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

## 4. Profiling Methods (ordered by preference for serving workloads)

### Method 1 (preferred): Framework-Native Profiling API

For serving frameworks (sglang, vLLM), use the built-in profiling endpoint.
This runs torch.profiler inside each TP worker process — no ptrace needed,
no OOM risk, produces per-rank traces.

```bash
# sglang: Start profiling (server must be running)
curl -X POST http://localhost:30000/start_profile

# Send your decode requests here...

# Stop profiling — traces saved to /tmp/ in the container
curl -X POST http://localhost:30000/stop_profile
```
Output: per-rank Chrome trace files (`.trace.json.gz`). Parse with the
trace JSON method in §3 above.

**Limitation**: torch.profiler may show AllReduce/communication as generic
"Other" and Triton kernels as `main_kernel`. If >40% of time is unidentified,
supplement with `rocprofv3` (Method 2/3) to resolve kernel names.

### Method 2: rocprofv3 --attach (for kernel-level resolution)

Attach to a running server process for HSA-layer kernel tracing.
Resolves Triton `main_kernel` names and shows AllReduce internals.

```bash
rocprofv3 --attach <SERVER_PID> --kernel-trace --hip-trace --attach-duration-msec 15000
```
- Requires container with `--cap-add SYS_PTRACE`
- May not produce output for multi-process TP servers (ptrace limitation)
- Best for single-process scripts or when Method 1 gives opaque kernel names

### Method 3: rocprofv3 wrap-launch (for standalone scripts)

```bash
rocprofv3 --hip-trace --kernel-trace -d /tmp/profile_output/ python3 script.py
# Output: hip_api_trace.csv, hip_activity_trace.csv, kernel_trace.csv

# Aggregated kernel stats (fastest way to get top-N hotspots)
rocprofv3 --stats python3 script.py
```
- May OOM on large models — use OOM workarounds below
- Best for microbenchmarks or single-GPU profiling

### Large Model OOM Workarounds (for Method 2/3)

When profiling large models (e.g. GLM-5.1-FP8 TP=8), rocprofv3 tracing
buffers need extra VRAM. If profiling OOMs:

1. **Use Method 1** (framework-native API) — no extra VRAM needed
2. **Reduce KV cache**: `--mem-fraction-static 0.5` (frees VRAM for profiler)
3. **Disable CUDA graphs**: `--disable-cuda-graph` (graph capture needs extra VRAM)
4. **Shorten sequences**: Reduce max sequence length (e.g. `--max-total-tokens 2048` for sglang)
5. **Profile a microbench**: Write a minimal script that loads the model and runs
   1 decode step — don't profile the full serving stack
6. **Single-GPU profiling**: `ROCR_VISIBLE_DEVICES=0` to profile one rank only

**Container requirement**: When using `--attach` mode, the container MUST be
launched with `--cap-add SYS_PTRACE` for ptrace access.

### Triton `main_kernel` Opaque Naming

Triton compiles all kernels to `main_kernel` variants. Three ways to identify them:

1. **rocprofv3 kernel-trace** (primary): HSA-layer names include hash/function suffixes
2. **Triton cache inspection**: `ls ~/.triton/cache/` → `cat ~/.triton/cache/<hash>/metadata.json`
3. **Autotuning log**: `TRITON_PRINT_AUTOTUNING=1 python3 script.py 2>&1 | grep kernel`

## 5. Decode vs Prefill: Profile Separately

Prefill and decode have different hotspots:
- **Prefill**: Large M → GEMM-dominated → CK/hipBLAS may be optimal
- **Decode**: Small M (= batch_size) → memory-bound → Triton small-M kernels may be better

```bash
# Prefill-dominated workload
python3 bench.py --num-prompts 1 --input-len 1024 --output-len 1

# Decode-dominated workload
python3 bench.py --num-prompts 1 --input-len 16 --output-len 512
```

Use `roctx` markers (ROCm equivalent of nvtx) to tag phases in code.
Simplest approach — PyTorch's nvtx wrapper maps to roctx on ROCm:
```python
# Preferred: PyTorch nvtx wrapper (auto-maps to roctx on ROCm)
torch.cuda.nvtx.range_push("decode_step")
# ... decode forward pass ...
torch.cuda.nvtx.range_pop()

# Alternative: direct roctx ctypes (if PyTorch nvtx not available)
import ctypes
roctx = ctypes.CDLL("libroctx64.so")
roctx.roctxRangePush.argtypes = [ctypes.c_char_p]
roctx.roctxRangePush.restype = ctypes.c_int
roctx.roctxRangePop.argtypes = []
roctx.roctxRangePop.restype = ctypes.c_int
roctx.roctxRangePush(b"decode_step")
# ... decode forward pass ...
roctx.roctxRangePop()
```

## 6. Profiling Failure = Blocker, Not Footnote

If profiling fails (OOM, tool not found, graph-obscured, opaque kernel names):
- **Do NOT proceed to blind tuning.**
- First fix profiling using the OOM workarounds above.
- If >40% of GPU time is "Other" / unidentified, escalate this as a blocker
  and try `rocprofv3 --kernel-trace` or Triton cache inspection.

## 7. Required Profiling Artifacts

After profiling, you MUST write these to `optimization_state.json`:
- `profiler_command`: exact command used
- `profile_output_path`: path to the produced artifact
- `profiling_scope`: `decode_fullstack` | `prefill_only` | `layer_microbench` | `reduced_memory_decode`
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
| **Triton** | `main_kernel` (use rocprofv3 to resolve) | Backend switching, autotuning |

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
