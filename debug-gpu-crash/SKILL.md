---
name: debug-gpu-crash
description: Debug GPU crashes (HIP/ROCm errors) in SGLang on AMD MI300X/MI355X using the @debug_kernel_api logging decorator. Use when a kernel crashes with hipErrorIllegalAddress, hipErrorAssert, device-side assert, out-of-bounds, NaN/Inf, or HIP stream capture failures. AMD counterpart of debug-cuda-crash.
---

# Skill: Debugging GPU Crashes with Kernel API Logging (AMD HIP/ROCm)

**When to use:** a SGLang kernel crashes on AMD MI300X (gfx942) / MI355X (gfx950) with errors like:
- `hipErrorIllegalAddress` / `HIP error: illegal memory access`
- `hipErrorAssert` / `device-side assert triggered`
- `hipErrorInvalidValue` / `out-of-bounds`
- `NaN` / `Inf` in tensor outputs
- `hipErrorLaunchFailure` / `HIP graph capture failed`
- ROCm `wavefront` traps, `SGPR`/`VGPR` overflows
- MIOpen / hipBLAS / AITER kernel failures

**Target hardware:** AMD MI300X (gfx942, CDNA3, 8 HBM3 stacks, 192GB) and MI355X (gfx950, CDNA4, 256GB). Most advice is generic; gfx942 vs gfx950 differences are explicitly marked.

---

## Golden rule

**Capture inputs BEFORE the crash, then diagnose. Never guess.**

HIP errors often crash the program before normal debugging output is flushed. SGLang's `@debug_kernel_api` decorator logs inputs before execution, so you can still see what caused the crash even after the program aborts.

---

## Step 1: Enable Kernel API Logging

The `@debug_kernel_api` decorator is vendor-agnostic — it works on ROCm the same way as CUDA, because custom ops are registered through the same `register_custom_op(...)` / `register_custom_op_from_extern(...)` paths.

### Basic Logging (Function Names Only)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=1
export SGLANG_KERNEL_API_LOGDEST=stdout

python my_script.py
```

Output:
```
================================================================================
[2026-07-07 19:30:12] SGLang Kernel API Call: RMSNorm.forward
================================================================================
[2026-07-07 19:30:12] SGLang Kernel API Call: sglang.quant_method.UnquantizedLinearMethod.apply
================================================================================
[2026-07-07 19:30:12] SGLang Kernel API Call: sglang.custom_op.fused_inplace_qknorm
```

### Detailed Logging (Inputs with Metadata)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=2
export SGLANG_KERNEL_API_LOGDEST=file:/tmp/sglang_kernel_api.log
```

Level 2 captures tensor shapes, dtypes, strides, device, and (for small tensors) value summaries. Use `file:` destination for level 2 — stdout is too verbose.

---

## Step 2: Identify the Crashing Kernel

The log shows the LAST kernel API call before the crash — that's your suspect. Cross-check with the HIP error:

| HIP error | Likely kernel type | What to inspect |
|---|---|---|
| `hipErrorIllegalAddress` | attention / MoE scatter / quant | input tensor shapes, index tensors (out-of-range?), `seq_len` vs `max_seq_len` |
| `hipErrorAssert` | quantization (FP8/MXFP4) | input range (FP8 clamps at ±448, MXFP4 at ±6; NaN/Inf in input?) |
| `hipErrorInvalidValue` | rotary / RoPE | `position_ids` negative? `inv_freq` NaN? |
| `hipErrorLaunchFailure` | any kernel | usually a prior async error — check `hipGetLastError()` before launch |
| MIOpen `MIOPEN_STATUS_INVALID_VALUE` | conv / VAE | input layout (NDHWC vs NCDHW?), `channels_last_3d` flag? |
| AITER `aiter.common.exception` | fused MoE / allreduce | `SGLANG_USE_AITER=1` set? gfx942 vs gfx950 gate? |

---

## Step 3: ROCm-Specific Debugging Tools

### `rocm-smi` (GPU state)
```bash
rocm-smi                   # GPU utilization, memory, temp
rocm-smi --showmeminfo vram # VRAM usage per GPU
rocm-smi --showseats        # ECC errors (MI300X reports UMC errors here)
```

### `rocminfo` (hardware discovery)
```bash
rocminfo | grep -E "Name|Marketing Name|Compute Unit"  # confirm gfx942 vs gfx950
```

### `HIP_LAUNCH_BLOCKING=1` (sync kernel launches)
```bash
HIP_LAUNCH_BLOCKING=1 python my_script.py
```
Makes every HIP call synchronous — the error traceback now points at the exact kernel launch that failed, not a later async catch. Essential for `hipErrorIllegalAddress` triage.

### `AMD_LOG_LEVEL=4` (HIP API logging)
```bash
AMD_LOG_LEVEL=4 python my_script.py 2>&1 | grep -E "hip|HIP" | tail -50
```
Logs every HIP API call. Very verbose — pipe to file and grep for the last calls before crash.

### `MIOpen_ENABLE_LOGGING=1` (MIOpen kernel selection)
```bash
MIOpen_ENABLE_LOGGING=1 python my_script.py 2>&1 | grep -i "miopen\|conv\|ndhwc"
```
Shows which Conv kernel MIOpen dispatched (NDHWC vs NCDHW). Use when debugging LTX-2 VAE channels-last-3d (#50) or any conv perf issue.

---

## Step 4: Common AMD-Specific Crash Patterns

### Pattern A: FP8 quantization overflow on MI300X
MI300X (gfx942) FP8 (E4M3) clamps at ±448. If a tensor has values > 448, `hipErrorAssert` fires in the quantization kernel. The `@debug_kernel_api` level-2 log will show the input tensor's max value before the crash.

**Fix:** clamp inputs before quant, or use `amd` MXFP4 quantization (MI355X only, clamps at ±6 but with per-block scaling).

### Pattern B: gfx942 vs gfx950 kernel dispatch
AITER kernels gate on `_is_gfx95_supported` (gfx950 / MI355X). On MI300X (gfx942), these return `None` and fall back to unfused paths. If a kernel crashes on MI300X but works on MI355X, check the gfx gate:

```python
from sglang.srt.layers.communicator import _is_gfx95_supported
print(f"gfx95 supported: {_is_gfx95_supported}")  # False on MI300X
```

### Pattern C: MIOpen NDHWC Conv3d fallback
`channels_last_3d` (NDHWC) Conv3d may fall back to NCDHW internally on some MIOpen versions, causing a silent perf regression (not a crash). Check with `MIOpen_ENABLE_LOGGING=1`.

### Pattern D: ROCm 7.x vs 6.x container mismatch
MI300X/MI355X require ROCm 7.0+ and driver 6.14+. A ROCm 6.4 container gives "No HIP GPUs are available" inside the container even though `rocm-smi` works on the host. See [[project_rocm_base_image_fleet_invariant]].

### Pattern E: numa_balancing=1 hang on MI300X
MI300X with `numa_balancing=1` (default on some kernels) causes TP-wide deadlock during hicache alloc — looks like a hang, not a crash. Fix: `echo 0 > /proc/sys/kernel/numa_balancing`. See [[project_sglang_numa_binding_amd_gap]].

---

## Step 5: Capture for Post-Mortem

When a crash happens, capture these BEFORE killing the container (per [[feedback_dont_kill_failed_containers]]):

```bash
# 1. Container logs
docker logs <container> > /tmp/crash_container.log 2>&1

# 2. Inspect state (don't rm)
docker inspect <container> > /tmp/crash_inspect.json

# 3. rocm-smi snapshot
docker exec <container> rocm-smi > /tmp/crash_rocm_smi.txt

# 4. Kernel API log (if level 2 was enabled)
docker cp <container>:/tmp/sglang_kernel_api.log /tmp/

# 5. Core dump (if enabled)
docker cp <container>:/tmp/core /tmp/ 2>/dev/null
```

---

## Related skills

- [`debug-cuda-crash`](../debug-cuda-crash/) — NVIDIA counterpart (same `@debug_kernel_api` decorator, different error patterns)
- [`rocprof-report-skill`](../rocprof-report-skill/) — AMD profiler (replaces `ncu-report-skill`)
- [`amd-kernel-wiki`](../amd-kernel-wiki/) — AMD kernel optimization knowledge base (replaces `KernelWiki`)

---

## References

- SGLang `@debug_kernel_api` decorator: `python/sglang/srt/utils/debug_kernel_api.py`
- ROCm error codes: https://rocm.docs.amd.com/en/latest/reference/hip_runtime/api/error.html
- MIOpen logging: https://github.com/ROCm/MIOpen#logging
