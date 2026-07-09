---
name: rocprof-report-skill
description: Profile HIP/ROCm kernels with rocprof/omniperf on AMD MI300X (gfx942) / MI355X (gfx950). Use when the user asks to profile a kernel, analyze its performance, diagnose bottlenecks, read a rocprof/omniperf report, or write an optimization plan. AMD counterpart of ncu-report-skill.
---

# Skill: HIP Kernel Profiling (AMD MI300X/MI355X / rocprof + omniperf)

**When to use:** user asks to profile a HIP kernel, analyze its performance, find its bottlenecks, or write an optimization plan based on rocprof/omniperf data. Triggers include: "profile X", "why is this kernel slow", "rocprof report says...", "how to optimize next", "help me read this rocprof report".

**Target hardware:** AMD MI300X (gfx942, CDNA3, 8 HBM3 stacks, 192GB, 304 CUs) and MI355X (gfx950, CDNA4, 256GB, 320 CUs). Most advice is generic; gfx942 vs gfx950 differences are explicitly marked.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing HIP kernels are under-performing for exactly one reason that rocprof/omniperf can tell you in 10 seconds. Don't invent hypotheses before you have the report. Don't start coding a fix before you've matched the observed pattern to a known diagnosis. Don't write a wall of suggestions — rank them by evidence and expected impact.

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** under `profile/<run_name>/` at the repo root — **one directory per run**, never reuse an existing one. Each run contains its own `harness/`, `reports/`, `analysis/`, and `REPORT.md`.

1. **Decide what you're profiling.** What inputs? Which dispatch path? What question do you want answered? If the kernel takes variable-sized inputs, pick specific representative shapes from the user's workload — don't profile with arbitrary inputs.

2. **Build a standalone harness** unless the user is profiling through their existing binary. Harnesses compile in seconds, run the kernel in isolation, and let you map SASS back to source. Compile into `profile/<run_name>/harness/`.

3. **Run two profiles**:
   - `rocprof --stats` for the overview (kernel timings, API calls)
   - `omniperf profile -n <run_name> -- <binary>` for per-instruction stall attribution (the AMD equivalent of NCU's `--set source`)

4. **Parse with `omniperf` Python API** — not by eye-balling the CLI. Write analysis outputs to `profile/<run_name>/analysis/`.

5. **Work through the six analysis dimensions** (see below). Every one matters, but on any given kernel only 1–2 will dominate.

6. **Match patterns to the diagnosis playbook** (see below). It maps rocprof/omniperf signal → likely cause → concrete fix.

7. **Write the report** at `profile/<run_name>/REPORT.md` with evidence-backed recommendations, ranked by expected impact.

---

## Tool comparison: rocprof vs omniperf vs rocm-smi

| Tool | AMD equivalent of | What it's for | Output |
|---|---|---|---|
| `rocm-smi` | `nvidia-smi` | GPU state (util, mem, temp, ECC) | CLI table |
| `rocprof` (v2) | `nvprof` / `ncu --set full` | Kernel timings, API traces, HW counters | CSV / JSON / text |
| `omniperf` | `ncu --set source` | Per-instruction stall attribution, roofline analysis | Interactive HTML + CSV |
| `rocprof-compute` (newer) | `ncu` (full) | Unified profiling (replaces rocprof + omniperf in ROCm 7+) | HTML + CSV |

**Use `omniperf` for deep analysis** — it's the closest AMD equivalent to NCU's source-level stall attribution. Use `rocprof --stats` for quick "which kernels are slow" overviews.

---

## Collection recipes

### Quick overview (which kernels are slow?)
```bash
rocprof --stats -o profile/<run_name>/reports/stats.csv \
  python my_script.py
```

### Full HW counter profile
```bash
rocprof -o profile/<run_name>/reports/counters.csv \
  --stats \
  python my_script.py
```

### omniperf deep dive (per-instruction stalls)
```bash
omniperf profile -n <run_name> \
  --profile-mode=1 \
  -- python my_script.py

# Then analyze:
omniperf analyze -p workloads/<run_name>/ \
  --list-metrics gfx942  # or gfx950 for MI355X
```

### With kernel API logging (combine with debug-gpu-crash)
```bash
SGLANG_KERNEL_API_LOGLEVEL=2 \
SGLANG_KERNEL_API_LOGDEST=file:/tmp/kernel_api.log \
rocprof --stats -o profile/<run_name>/reports/stats.csv \
  python my_script.py
```

---

## Six analysis dimensions

### 1. Occupancy (waves per CU)
- **Metric:** `GRBM_GUI_ACTIVE` / `SQ_WAVES` (rocprof); `Occupancy` panel (omniperf)
- **MI300X:** 304 CUs, max 64 waves/CU = 19456 waves. A kernel launching < 304 wavefronts = <1 wave/CU = low occupancy.
- **MI355X:** 320 CUs.
- **Diagnosis:** low occupancy → consider increasing work per thread, or fusing kernels to amortize launch overhead.

### 2. Compute / memory balance (roofline)
- **Tool:** `omniperf analyze --roofline`
- **MI300X peak:** 1307 TF/s FP8 (matrix), 163 TF/s BF16 (vector); 5.3 TB/s HBM3.
- **MI355X peak:** 3154 TF/s FP8 (matrix), 197 TF/s BF16 (vector); 8 TB/s HBM3e.
- **Diagnosis:** if the kernel is below the roofline knee, it's latency-bound (not compute or memory bound). Check stalls (dimension 3).

### 3. Stall reasons (per-SIMD)
- **Metric:** `SQ_INSTS_*` (rocprof); `Stall reasons` panel (omniperf)
- **Key stalls on gfx942/gfx950:**
  - `SQ_WAIT_LGVM` — waiting on LDS (shared memory) or global memory load
  - `SQ_WAIT_VMENT` — waiting on vector memory
  - `SQ_WAIT_SGPR` — waiting on scalar register (often due to loop bounds)
  - `SQ_WAIT_EXP` — waiting on export (output)
- **Diagnosis:** `SQ_WAIT_LGVM` dominant → memory-latency-bound. `SQ_WAIT_SGPR` dominant → register pressure / loop-carried deps.

### 4. Matrix unit utilization (MFMA)
- **Metric:** `SQ_INSTS_VALU_MFMA` (rocprof); `Matrix Unit` panel (omniperf)
- **MI300X:** MFMA (Matrix Fused Multiply-Add) is the tensor-core equivalent. FP8 MFMA is the highest-throughput path.
- **MI355X:** adds MXFP4 MFMA (block-scaled FP4).
- **Diagnosis:** low MFMA utilization → not using tensor cores. Check if the kernel is using `hipBLAS` / `AITER` / `CK` (Composable Kernel) paths, or falling back to VALU (vector ALU).

### 5. Timeline (tail effects)
- **Tool:** `rocprof --trace` (generates Chrome trace JSON); or `omniperf analyze --time`
- **Diagnosis:** if utilization drops in the tail (last 10% of kernel runtime), it's load-imbalance on variable-length inputs. Common in attention kernels with variable seq lengths.

### 6. Memory hierarchy (L1/L2/HBM)
- **Metric:** `TA_*` (texture/L1), `TCP_*` (L2 cache), `TCC_*` (HBM channels) — rocprof counter names
- **MI300X:** 8 HBM3 stacks, 16 channels. `TCC_*` counters show per-channel traffic — imbalance indicates bad memory access patterns.
- **Diagnosis:** low L1 hit rate + high HBM traffic → memory-bound. Check if the kernel is coalescing accesses (wavefront-wide loads).

---

## Diagnosis playbook (pattern → cause → fix)

| Signal | Likely cause | Concrete fix |
|---|---|---|
| Low occupancy + high `SQ_WAIT_SGPR` | Register pressure | Reduce register usage (spill to LDS), or reduce workgroup size |
| `SQ_WAIT_LGVM` dominant, low HBM traffic | L1/LDS latency-bound | Increase L1 reuse (tile larger), or prefetch with `buffer_load` |
| `SQ_WAIT_LGVM` dominant, high HBM traffic | HBM-bandwidth-bound | Coalesce accesses, use LDS staging, reduce working set |
| Low MFMA utilization, high VALU | Not using tensor cores | Route through `hipBLAS` / `AITER` / `CK`; check `SGLANG_USE_AITER=1` |
| Low MFMA on FP8 (MI300X) | FP8 path not dispatched | Check `quantization=fp8` + gfx942 support in the kernel |
| Low MFMA on MXFP4 (MI355X) | MXFP4 path not dispatched | Check `quantization=mxfp4` + `_is_gfx95_supported` gate |
| Tail effect (utilization drops at end) | Load-imbalance on variable inputs | Pad inputs, or split into balanced workgroups |
| MIOpen Conv3d slow | NDHWC fallback to NCDHW | Check `MIOpen_ENABLE_LOGGING=1`; ensure `channels_last_3d` flag |

---

## gfx942 (MI300X) vs gfx950 (MI355X) differences

| Feature | MI300X (gfx942) | MI355X (gfx950) |
|---|---|---|
| FP8 (E4M3/E5M2) | Yes (MFMA) | Yes (MFMA) |
| MXFP4 (block-scaled FP4) | **No** | Yes (MFMA) |
| MXFP8 (block-scaled FP8) | **No** | Yes |
| AITER fused MoE (MXFP4) | **No** (gfx95 gate) | Yes |
| AITER allreduce fusion | Yes (no gfx95 gate) | Yes |
| Peak FP8 TF/s | 1307 | 3154 |
| HBM bandwidth | 5.3 TB/s (HBM3) | 8 TB/s (HBM3e) |

**Key gotcha:** `_is_gfx95_supported` gates the MXFP4 / fused-MoE AITER paths. MI300X (gfx942) returns `False` and falls back to unfused paths. The AITER allreduce fusion is NOT gfx95-gated (works on both).

---

## Critical lessons

1. **Always compile with `-g` (debug info) and `-lineinfo`.** Without `-lineinfo`, omniperf's source view is blank and you cannot do per-line stall analysis. For PyTorch inline kernels, build a standalone harness.

2. **`HIP_LAUNCH_BLOCKING=1` is the first tool to reach for.** It makes every HIP call synchronous — the error traceback points at the exact kernel launch that failed. Essential for `hipErrorIllegalAddress` triage.

3. **MI300X reports ECC errors via `rocm-smi --showseats`.** Deferred UMC errors can cause SVM ioctl stalls during hicache alloc — looks like a hang. See [[project_n15_33_gpu_ecc_fault]].

4. **`numa_balancing=1` causes TP-wide deadlock on MI300X.** Not a perf issue — a hang. Always check `cat /proc/sys/kernel/numa_balancing` (should be 0). See [[project_sglang_numa_binding_amd_gap]].

5. **Don't delegate understanding.** Run the profiles yourself, open the reports, cite specific metric values. Never write "the profile shows it's memory-bound" — instead, name the two or three metric values that back your conclusion (e.g., "`TCC_HIT` rate under 10%, `SQ_WAIT_LGVM` stalls dominate, so the kernel is **HBM-latency-bound**, not compute-bound"). Fill in the actual numbers from your report. Specificity is the deliverable.

---

## Related skills

- [`ncu-report-skill`](https://github.com/DongyunZou/ncu-report-skill) — NVIDIA counterpart (Nsight Compute on B200/H100)
- [`debug-gpu-crash`](../debug-gpu-crash/) — AMD crash debugging (uses `@debug_kernel_api`)
- [`amd-kernel-wiki`](../amd-kernel-wiki/) — AMD kernel optimization knowledge base
- [`llm-torch-profiler-analysis`](../llm-torch-profiler-analysis/) — higher-level torch.profiler triage (vendor-agnostic, works on ROCm)

---

## References

- rocprof docs: https://rocm.docs.amd.com/en/latest/develop/development-tools/rocprofiler.html
- omniperf docs: https://rocm.docs.amd.com/en/latest/develop/development-tools/omniperf.html
- ROCm performance counters: https://rocm.docs.amd.com/en/latest/develop/development-tools/rocprofiler.html#performance-counters
- MI300X architecture: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- MI355X architecture: https://www.amd.com/en/products/accelerators/instinct/mi355/mi355x.html
