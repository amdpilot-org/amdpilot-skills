---
name: flydsl-lds-optimization
description: >
  Optimize LDS (shared memory) usage in FlyDSL kernels on AMD ROCm. Covers
  smem_allocator, bank-conflict patterns on AMD's 64-lane wavefront × 16-bank
  × 4-byte LDS, layout swizzling, and verification via rocprof. Use when
  profiling shows LDS stalls, you're writing a kernel with >8 KB LDS, or you
  need to swizzle to avoid bank conflicts in the MoE gate/up / down paths.
---

# FlyDSL LDS Optimization

AMD GPUs have **32 LDS banks, 4 bytes each** (as exposed to a wavefront of 64
lanes). Two lanes hitting the same bank at the same address (broadcast) is
fast. Two lanes hitting the same bank at **different** addresses is a bank
conflict and serializes.

## 1. smem_allocator — always use it

FlyDSL tracks LDS via a per-kernel `smem_allocator`:

```python
@flyc.kernel
def my_kernel(A: fx.Tensor, …):
    sA = fx.smem_allocator.alloc(BLOCK_M, BLOCK_K, dtype=A.dtype)
    sB = fx.smem_allocator.alloc(BLOCK_K, BLOCK_N, dtype=B.dtype)
    # sA + sB automatically packed into the block's LDS region
```

Do NOT hand-manage LDS offsets. The allocator handles alignment and avoids
off-by-one overlaps.

Capacity on MI355X: **64 KB per compute unit** (same as MI300X). A block with
`BLOCK_M=128, BLOCK_K=64` in bf16 uses `128*64*2 = 16 KB` for A + same for B =
32 KB. Leaves room for double-buffered prefetch.

## 2. Bank-conflict math

Given `LDS layout = (Shape, Stride)` and a per-thread access pattern, two
lanes conflict when:

```
(lane_a_offset % 32) == (lane_b_offset % 32)  AND
lane_a_offset != lane_b_offset
```

A single FP32 load per lane across 32 consecutive lanes: stride=1 → no
conflict (each lane hits a different bank).
A single BF16 load per lane across 32 consecutive lanes: stride=1 (in bf16
elements) = stride=0.5 (in banks) → 2 lanes per bank → 2-way conflict.

Workaround: vectorize the load (128-bit per lane = 8 bf16 = 2 bank-sets) OR
swizzle the LDS layout (§3).

## 3. Swizzling for bank-conflict avoidance

FlyDSL's `fx.make_swizzle(B, M, S)` encodes the CuTe-style XOR swizzle:

| Parameter | Meaning |
|---|---|
| `B` | Log2(bytes-per-swizzle-unit) — typically 3 for 8-byte units |
| `M` | Log2(number of bits XORed) — typically 3 for 8-byte stride |
| `S` | Log2(swizzle period) — typically 3 |

The common `(3, 3, 3)` swizzle rotates each LDS row so that sequential threads
hit different banks:

```python
swizzle = fx.make_swizzle(B=3, M=3, S=3)
base_layout = fx.make_layout(fx.make_shape(BLOCK_M, BLOCK_K), fx.make_stride(BLOCK_K, 1))
swizzled_layout = fx.composition(swizzle, base_layout)
sA = fx.smem_allocator.alloc(layout=swizzled_layout, dtype=fx.Dtype.bf16)
```

When to use which swizzle:

| Access pattern | Swizzle |
|---|---|
| GEMM A-fragment load (MxK, load K-contiguous) | `(3, 3, 3)` over the K axis |
| GEMM B-fragment load (KxN, load N-contiguous) | `(3, 3, 3)` over the N axis |
| Reduction across threads in a warp | `(2, 2, 3)` — narrower swizzle for warp-local |
| Store (register → LDS) for post-MMA | `(3, 3, 3)` matching MMA C-fragment layout |

If unsure, start with `(3, 3, 3)` over whichever axis you stride across per
thread — this is the common sweet spot on AMD for 8-byte per-thread accesses.

## 4. Padding as an alternative to swizzling

Quick and dirty: pad the column dimension to move bank indices apart.

```python
# 16 KB bf16 tile, pad 16 half-elements of padding per row
sA_layout = fx.make_layout(fx.make_shape(BLOCK_M, BLOCK_K + 16), ...)
```

Cost: wasted LDS bytes (16 columns × BLOCK_M × 2 B). Usually worse than
swizzling unless you need a specific swizzle period the FlyDSL helpers don't
expose.

## 5. Double-buffered prefetch

See [flydsl-kernel-authoring](../flydsl-kernel-authoring/SKILL.md) §3 for the
authoring idiom. The LDS budget doubles:

```
LDS per block ≈ 2 × (BLOCK_M × BLOCK_K × sizeof(A) + BLOCK_K × BLOCK_N × sizeof(B))
             + <MMA accumulator fragments live in registers, not LDS>
```

On MI355X with 64 KB LDS budget and BF16: `BLOCK_M=128, BLOCK_K=64, BLOCK_N=128`
→ 2 × (16 + 16) KB = 64 KB. That's exactly at the ceiling; leave margin (~4 KB)
for FlyDSL's own bookkeeping.

## 6. Verifying LDS usage and conflicts

### Compile-time LDS size

```bash
FLYDSL_DUMP_IR=1 /opt/venv/bin/python3 your_test.py 2>&1 | grep -i "shared"
```

Look for `llvm.mlir.addressof @__shared_mem` and the associated `bytes:` attr.

### Runtime bank-conflict count (rocprof)

```bash
rocprofv2 --kernel-filter 'flydsl_.*' \
          --counters LDSBankConflict \
          /opt/venv/bin/python3 your_bench.py
```

`LDSBankConflict > 0.1 * LDS_access_count` is a problem. `< 0.01x` is fine.

### Occupancy from LDS

Reduce per-block LDS → more concurrent blocks per CU → higher occupancy. Check
via:

```bash
rocprofv2 --counters Occupancy your_bench.py
```

If occupancy is <40% and LDS-per-block is >32 KB, LDS is the limiter — reduce
BLOCK_M or BLOCK_K.

## 7. Register pressure interplay

LDS and registers both contribute to block-per-CU capacity:

```
blocks_per_CU = min(
    regs_budget / regs_per_thread / threads_per_block,
    lds_budget / lds_per_block,
    max_blocks_per_CU_arch_cap,
)
```

MI355X: 65536 scalar regs + 512 vector regs per CU, 64 KB LDS per CU, max
32 blocks/CU. Optimize whichever is the binding constraint; if blocks-per-CU
≥ 4 already, LDS reduction buys nothing.

## 8. MoE-specific LDS patterns

In the K2.5 MoE stage1 kernel:

- **Gate/up are interleaved** in LDS — a single row of LDS holds gate[i] and
  up[i] adjacent so the fused `silu(gate) * up` issues one LDS load.
- **Topk weights stay in registers**, not LDS — they're tiny (`topk=8`) and
  per-token, so fetching to a per-lane register and broadcasting is cheaper.
- **Expert weights paged** — only one expert's weights live in LDS at a time,
  with the topk scheduler rearranging tokens so consecutive blocks hit the
  same expert (LDS reuse).

## 9. Do NOT do

- Use `__shared__` or raw HIP LDS intrinsics — FlyDSL layouts won't know about
  them. Always go through `smem_allocator`.
- Swizzle at compile time without verifying via rocprof. A mis-swizzle can
  *add* bank conflicts.
- Run multiple `smem_allocator.alloc` calls with dynamically-computed sizes —
  the allocator requires static sizes for correct LDS packing.
- Assume MI355X has 2x the LDS of MI300X. Both are 64 KB per CU.
