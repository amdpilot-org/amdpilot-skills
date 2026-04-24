# MoE Kernel Config Tuning on AMD ROCm

## Overview

The `fused_moe_triton` kernel in sglang uses JSON config files to control Triton block sizes, warps, and stages. Missing or default configs cause severe performance degradation (up to 3x slower). This guide covers the **correct workflow** for tuning these configs.

## MANDATORY Workflow: Profile → Analyze → Tune → Verify

**Do NOT skip to tuning. You must profile first.**

### Step 1: Profile with torch.profiler

Run `torch.profiler` (or your framework's built-in profiling endpoint) on the
benchmark, dump a Chrome trace, and parse it to see which kernels dominate GPU
time. The workflow is in the `gpu-profiling` skill — summary:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # run a few iterations of the benchmark
    ...
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("/tmp/trace.json")
```

Then aggregate by kernel name from the trace JSON to get a percent
breakdown.

**What to look for:**
- Which kernel takes the most GPU time (should be `fused_moe_kernel` at 80%+)
- How many calls per kernel (more calls = more launch overhead)
- Average time per call (high = compute-bound, low with many calls = launch-bound)

### Step 2: Analyze per-batch-size behavior

The MoE kernel behaves very differently at different batch sizes because `tokens_per_expert = M * topk / E` changes:

| M (batch) | tokens_per_expert (topk=8, E=384) | Behavior |
|-----------|----------------------------------|----------|
| 512 | ~10 | Very few tokens per expert → kernel is **launch-overhead-bound** |
| 2048 | ~43 | Moderate → transition zone |
| 4096 | ~85 | Good occupancy → **compute-bound** |
| 8192+ | ~170+ | High occupancy → compute-bound, benefits from large blocks |

**This is why one config doesn't fit all batch sizes.** Small M needs small BLOCK_SIZE_M (16-32) and few warps (1-2). Large M needs large BLOCK_SIZE_M (64-128) and more warps (4-8).

### Step 3: Understand the config lookup mechanism

Read `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`:

```python
# Config lookup order:
# 1. Search for file matching E, N, device_name, dtype
# 2. Filename format: E={E},N={N},device_name={device},dtype={dtype}.json
# 3. Inside the JSON: keys are batch sizes (as strings), values are config dicts
# 4. At runtime: find closest batch size key to actual M
```

**Important:** `device_name` is empty inside Docker containers (ROCm bug #5992). Config filenames use `device_name=` (empty).

Two separate config files are needed:
- **Up projection**: `E=384,N=128,device_name=,dtype=int4_w4a16.json`
- **Down projection**: `E=384,N=128,device_name=,dtype=int4_w4a16_down.json` (different access patterns)

### Step 4: Systematic tuning

Use the existing tuning infrastructure in `benchmark/kernels/fused_moe_triton/`:

```bash
# Option A: Use the tuning script (if int4_w4a16 support is added)
cd /workspace/sglang
/opt/venv/bin/python3 benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
  --dtype int4_w4a16 --num_experts 384 --shard_intermediate_size 2048 --tp_size 8

# Option B: Write a targeted search script for specific batch sizes
```

**Config parameter guide based on profiling:**

| Parameter | Compute-bound (large M) | Launch-bound (small M) |
|-----------|------------------------|----------------------|
| BLOCK_SIZE_M | 64-128 | 16-32 |
| BLOCK_SIZE_N | 64-128 | 32-64 |
| BLOCK_SIZE_K | 64-128 | 64 |
| GROUP_SIZE_M | 8 | 1-4 |
| num_warps | 4-8 | 1-2 |
| num_stages | 2 | 2 |

**Always benchmark each config.** Never guess — a config that looks good on paper can be 2x slower due to shared memory pressure or register spilling.

### Step 5: Verify with the test harness

```bash
cd /workspace/sglang && /opt/venv/bin/python3 /workspace/test_harness.py
```

Check per-batch-size speedups. If small M (512, 2048) shows <1.3x speedup, you need different configs for those sizes. The geometric mean across all batch sizes determines the Tier 3 score.

## Common Mistakes

1. **Skipping profiling and guessing configs** — You'll plateau at ~1.4x. Profiling shows you exactly where the bottleneck is (compute vs memory vs launch overhead) and guides config choice.

2. **Using one config for all batch sizes** — Small and large batch sizes need fundamentally different configs. Check per-batch speedup, not just the geometric mean.

3. **Not separating UP and DOWN projections** — They have different memory access patterns. Tune them separately.

4. **Fabricating configs without benchmarking** — A config with BLOCK_SIZE_M=256 might seem fast but can crash due to shared memory limits. Always benchmark.

5. **Not profiling again after tuning** — Re-run `torch.profiler` after applying configs to verify the target kernel actually sped up. Variance can fool you.
