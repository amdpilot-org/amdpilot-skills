---
name: flydsl-kernel-authoring
description: >
  Author new FlyDSL kernels using the @flyc.kernel / @flyc.jit decorators,
  fx.Tensor / fx.Constexpr / fx.Stream types, launch grid specification, and
  fx.block_idx / fx.thread_idx indexing. Use when writing a new fly dialect
  kernel from scratch, prototyping a replacement for a Triton kernel, or
  extending the AITER kimi-K2.5 MoE path with a new shape.
---

# Authoring FlyDSL Kernels

Style reference: FlyDSL upstream `examples/` + `kernels/moe_gemm_2stage.py`
live at `/opt/FlyDSL/python/kernels/` in the K2.6 eval image. Read those before
writing anything new.

## 1. Minimal @flyc.kernel skeleton

```python
import flydsl
from flydsl import flyc, fx
import torch

@flyc.kernel
def vec_add(
    a: fx.Tensor,           # device tensor; shape / dtype inferred at launch
    b: fx.Tensor,
    c: fx.Tensor,
    n: fx.Constexpr[int],   # compile-time constant
):
    tid = fx.block_idx.x * 256 + fx.thread_idx.x
    if tid < n:
        c[tid] = a[tid] + b[tid]
```

Launch it:

```python
N = 1024
a, b = torch.randn(N, device="cuda"), torch.randn(N, device="cuda")
c = torch.empty_like(a)
stream = torch.cuda.current_stream()
vec_add.launch(
    grid=((N + 255) // 256, 1, 1),
    block=(256, 1, 1),
    stream=fx.Stream(stream.cuda_stream),
)(a, b, c, N)
```

## 2. @flyc.kernel vs @flyc.jit

| Decorator | When to use |
|---|---|
| `@flyc.kernel` | Pure GPU kernel. Must call via `.launch(grid=…, block=…)(args)` |
| `@flyc.jit` | Device helper / inline function called from inside `@flyc.kernel`. No launch. |

Nest helpers via `@flyc.jit`:

```python
@flyc.jit
def silu(x):
    return x / (fx.Constexpr(1.0) + fx.exp(-x))

@flyc.kernel
def mlp_fused(gate: fx.Tensor, up: fx.Tensor, out: fx.Tensor, n: fx.Constexpr[int]):
    tid = fx.block_idx.x * 256 + fx.thread_idx.x
    if tid < n:
        out[tid] = silu(gate[tid]) * up[tid]
```

## 3. Type system cheatsheet

| Type | Purpose |
|---|---|
| `fx.Tensor` | Device tensor (any dtype). Dtype/shape bound at launch. |
| `fx.Constexpr[int]` / `fx.Constexpr[float]` | Compile-time constant (kernel gets recompiled per distinct value — use sparingly) |
| `fx.Stream` | HIP stream wrapper — required by `.launch(stream=...)` |
| `fx.block_idx.x/y/z` | Block index inside grid |
| `fx.thread_idx.x/y/z` | Thread index inside block |
| `fx.block_dim.x`, `fx.grid_dim.x` | Block / grid dimensions |
| `fx.Dtype.bf16 / f16 / f32 / i32 / u8 / i4` | Dtype enum |

## 4. Launch grid pattern

Always compute grid inside Python host code, not inside the kernel:

```python
BLOCK_M, BLOCK_N = 128, 128
grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, 1)
block = (256, 1, 1)  # 256 threads = 4 wavefronts × 64 lanes on AMD
kernel.launch(grid=grid, block=block, stream=stream)(…)
```

Default wavefront on AMD is **64 lanes** (not 32 as on NVIDIA). A `block=(64,…)`
is one wavefront; `block=(256,…)` is 4 wavefronts — the common sweet spot for
MFMA-heavy kernels.

## 5. Reusing FlyDSL layout algebra

For anything more structured than elementwise, drop into layout algebra. See
[flydsl-tile-programming](../flydsl-tile-programming/SKILL.md) for the primitives
(`make_layout`, `make_tiled_copy`, `partition_S / partition_D`). Don't hand-roll
tile loops — the composition operators generate the same code with less bug
surface.

## 6. Numerical safety

- **bf16 rounding** — use `fx.cast(x, fx.Dtype.f32)` before an accumulation,
  then cast back. Naive bf16 accumulation drifts after ~256 terms.
- **Denormals** — FlyDSL does NOT flush subnormals by default. Enable FTZ via
  the MFMA atom config (see [flydsl-gemm-optimization](../flydsl-gemm-optimization/SKILL.md))
  if your workload saturates near zero.
- **W4A16 unpack** — use the built-in `fx.unpack_i4` / `fx.cast_to(bf16)`
  helper, not a manual shift-and-mask. The helper emits optimized MFMA-friendly
  layouts.

## 7. Debug printing

FlyDSL has no `printf` inside kernels. To debug numerics:

```python
# 1. Write intermediate to a scratch tensor, inspect on host:
scratch = torch.empty(M, N, device="cuda")
my_kernel.launch(grid, block)(a, b, c, scratch)  # kernel writes intermediate to scratch
print(scratch[:4, :4])  # CPU side

# 2. Or use the compare-mode env vars in AITER (see flydsl-debug-kernel)
```

Do NOT add `print` statements to the Python function body — they fire at
trace time, not runtime, and emit nothing useful.

## 8. When to write a new kernel vs tune existing

| Scenario | Decision |
|---|---|
| Existing kernel in `kernels/` matches your shape family | Tune/parametrize it (M/N/K blocks, tiled copy widths) |
| Same compute, different dtype | Add a dtype dispatch in the existing kernel |
| New fused op not in FlyDSL upstream | Write a new `@flyc.kernel`. See `examples/04-preshuffle_gemm.py` for the pattern. |
| Exotic layout Triton can't express | FlyDSL layout algebra is the tool; write new kernel. |
| Single-GEMM performance | Use aiter tuned GEMM via `torch.ops.aiter.*` — not FlyDSL. Authoring time dominates for one-shot GEMMs. |

## 9. Contract with AITER fused-MoE dispatcher

AITER's `dev/kimi-K2.5` branch calls into FlyDSL through:

```python
# inside aiter/ops/fused_moe/*.py
from aiter.ops.fused_moe.flydsl_moe_stage1 import stage1_impl
out = stage1_impl(hidden_states, gate_weight, up_weight, topk_ids, topk_weights, ...)
```

Your new kernel must match that call signature if you're adding a new shape
family to the MoE stage1 path. See
[flydsl-gemm-optimization](../flydsl-gemm-optimization/SKILL.md) §5 for the
contract details.

## 10. Do NOT do

- Mutate `fx.Constexpr` values at runtime — they trigger a recompile per
  distinct value and will cache-spam `~/.flydsl/cache/`.
- Use Python `for i in range(n)` in `@flyc.kernel` when `n` is a runtime value
  — the tracer can't unroll and you'll hit an opaque MLIR error. Use
  `fx.range` or a `fx.while_loop`.
- Call torch ops from inside `@flyc.kernel`. Pre-compute host-side tensors and
  pass them in.
- Assume `block = (256, 1, 1)` is optimal for every shape. Sweep (128, 256,
  512) once per shape family; bigger is NOT always better on AMD's 64-lane
  wavefronts.
