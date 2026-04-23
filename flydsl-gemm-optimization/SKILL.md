---
name: flydsl-gemm-optimization
description: >
  Optimize GEMM and fused-MoE kernels in FlyDSL on AMD ROCm. Covers MFMA atoms
  per-arch (gfx942 vs gfx950), tiled MMA, split-K, the preshuffle GEMM recipe,
  the Kimi-K2.5 MoE 2-stage GEMM mapping (kernels/mixed_moe_gemm_2stage.py),
  and per-shape atom selection. Use when writing or tuning a FlyDSL GEMM,
  especially for the Kimi-K2.6 W4A16 MoE path on MI355X.
---

# FlyDSL GEMM & MoE Optimization

**Prerequisite:** read [flydsl-tile-programming](../flydsl-tile-programming/SKILL.md)
for layout algebra primitives. This skill assumes you know `make_tiled_copy`,
`partition_S / D`, `smem_allocator`.

## 1. MFMA atoms per AMD architecture

MFMA (Matrix Fused Multiply-Add) is the AMD hardware MMA instruction. Different
architectures expose different atoms. FlyDSL encodes them as
`fx.MFMAAtom.<name>`:

| Arch | Atom (BF16) | Atom (FP16) | Atom (FP8) | Notes |
|---|---|---|---|---|
| gfx942 (MI300X) | `MFMA_F32_32x32x8_BF16` | `MFMA_F32_32x32x8_F16` | `MFMA_F32_16x16x32_FP8` | v1 MFMAs |
| gfx950 (MI355X) | `MFMA_F32_32x32x16_BF16` | `MFMA_F32_32x32x16_F16` | `MFMA_F32_16x16x64_FP8` | v2 MFMAs, 2x K-per-op |

The "k-per-op" is the number of reduction lanes consumed per MFMA issue —
higher is better for compute density. gfx950's v2 atoms halve the number of
MFMA issues for the same K reduction vs gfx942.

W4A16 path uses the BF16 atom after unpacking:

```python
# Inside @flyc.kernel:
# 1. Load packed int4 from global
w_packed = fx.load(sB_packed, copy_atom=UniversalCopy32b)
# 2. Unpack to bf16
w_bf16 = fx.unpack_i4_to_bf16(w_packed, scale)
# 3. MFMA accumulates in fp32
acc = fx.mfma(MFMA_F32_32x32x16_BF16, a_bf16, w_bf16, acc)
```

## 2. Tiled MMA

The tiled-MMA partitions MFMA atoms across threads of a block:

```python
tiled_mma = fx.make_tiled_mma(
    atom=fx.MFMA_F32_32x32x16_BF16,            # per-arch choice
    atom_layout=fx.make_layout(fx.make_shape(2, 2)),   # 2x2 tiling of atoms
    permutations=fx.make_layout(fx.make_shape(1, 1)),  # no swap
)

# Inside kernel:
thr_mma = tiled_mma.get_slice(tid)
tCrA = thr_mma.partition_A(sA)   # A-fragment register tile
tCrB = thr_mma.partition_B(sB)   # B-fragment register tile
tCrC = thr_mma.partition_C(rC)   # accumulator register tile

for k_block in fx.range(0, num_k_blocks):
    tCrC = fx.gemm(tiled_mma, tCrA[:, :, k_block], tCrB[k_block, :, :], tCrC)
```

A tiled-MMA's tile size defines the smallest unit of work (one "MMA call").
Pick `atom_layout` so BLOCK_M x BLOCK_N is evenly divisible.

## 3. Split-K for small-M / decode shapes

When M is small (decode phase, BS=1–16), a single block doesn't fill the K
dimension. Split-K creates multiple blocks along K and reduces:

```python
@flyc.kernel
def gemm_splitk(...):
    kblock_idx = fx.block_idx.z   # split-K along z-grid axis
    k_start = kblock_idx * K_PER_BLOCK
    k_end = min(k_start + K_PER_BLOCK, K)
    # ... accumulate locally, write partial to global
    fx.atomic_add(partial_out, local_acc)  # atomic over kblocks
```

Or use `fx.split_k` helper which emits the atomic-reduce pattern for you.

Split-K is **crucial** for the BS=1 decode path that PR #23381 tunes — without
it, K-large shapes (moe inter_dim=7168) are memory-bound even on MI355X.

## 4. Preshuffle GEMM recipe (from `examples/04-preshuffle_gemm.py`)

Weight "preshuffling" rearranges the weight tensor so that the natural HBM →
LDS copy pattern matches the MFMA operand layout without a post-copy shuffle
in LDS. Trade: one-time host-side weight permutation for zero LDS shuffle cost.

```python
# Host side (one-time, before serving starts):
W_preshuffled = fx.preshuffle_weight(
    W, atom=fx.MFMA_F32_32x32x16_BF16, mode="B",
)

# Kernel side: the copy atom + MMA atom now match:
tiled_copy_b = fx.make_tiled_copy(
    atom=fx.UniversalCopy128b,    # 16B per lane
    src_layout=...,               # matches MFMA B-fragment shape
    dst_layout=...,
)
```

For AITER K2.5 MoE, preshuffling already happens at model-load time — do NOT
re-preshuffle at inference time.

## 5. Kimi-K2.5 MoE 2-stage GEMM — the blog kernel

Source: `/opt/FlyDSL/python/kernels/mixed_moe_gemm_2stage.py` (or upstream
`kernels/moe_gemm_2stage.py`).

### Stage 1: gate/up projection

```
Input:  hidden_states (tokens, hidden_dim)
Weights: W_gate_up (E, 2*inter_dim, hidden_dim)   # packed
Dispatch: topk_ids (tokens, topk)
Output: gated_out (tokens*topk, inter_dim)         # activation applied fused
```

Key design choices in the K2.5 kernel:
- **Topk-aware tile scheduler** — each block is assigned a contiguous range
  of (expert, token*topk) pairs so consecutive blocks touch the same expert's
  weights, reusing LDS.
- **Fused SiLU + multiply** — the gate and up columns are interleaved; the
  kernel computes `silu(gate) * up` before writing back.
- **W4A16 path** — weights packed as int4, scale tensor at group granularity
  (group_size=128 is the default for Kimi-K2.6).

### Stage 2: down projection

```
Input:  gated_out (tokens*topk, inter_dim)
Weights: W_down (E, hidden_dim, inter_dim)
Output: reduced (tokens, hidden_dim)   # sum-reduced across topk
```

- **Reduce-across-topk fused** — Stage 2 also reduces across the `topk`
  dimension for the same input token, eliminating a post-op reduction.
- **`FLYDSL_W4A16_HYBRID=w2_bf16`** — keeps W4A16 on Stage 1 but promotes
  Stage 2 to BF16 weights. Trades memory (~1.5x the Stage-2 footprint) for
  numerical stability; blog reports this is the sweet spot.

### Entry point in AITER

```python
# /sgl-workspace/aiter/aiter/ops/fused_moe/flydsl_moe_stage1.py
def stage1_impl(
    hidden_states: Tensor,        # (tokens, hidden_dim)
    w_gate_up: Tensor,            # (E, 2*inter_dim, hidden_dim)
    w_gate_up_scale: Tensor,      # (E, 2*inter_dim, hidden_dim//128)
    topk_ids: Tensor,             # (tokens, topk)
    topk_weights: Tensor,         # (tokens, topk)
    *,
    activation: str = "silu",
    dtype_out: torch.dtype = torch.bfloat16,
) -> Tensor:
    ...
```

When adding a new shape family, match this signature exactly.

## 6. Per-shape atom selection heuristic

Given MoE shape `(tokens, hidden_dim, inter_dim, E, topk)`:

| Condition | Recommended tile shape |
|---|---|
| tokens ≥ 16384 (prefill-heavy) | `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64`, v2 MFMA 32x32x16 |
| tokens ≤ 512 (decode-heavy) | `BLOCK_M=32, BLOCK_N=128, BLOCK_K=128` + split-K=4 |
| inter_dim is small (512) | Favor larger BLOCK_N; BLOCK_N ≥ inter_dim if possible |
| W4A16 | Keep BLOCK_K ≥ 128 — int4 unpacking amortizes better |
| BF16 | BLOCK_K = 64 is usually sufficient |

These are **starting points**. Sweep in a narrow range; do NOT binary-search
over 20+ combinations per shape — that's autotune territory and violates the
"one scored source edit per trial" rule.

## 7. Verification workflow

Before reporting a win:

1. Unit-test the kernel on the target shape with `torch.testing.assert_close`
   (rtol=1e-2 for bf16).
2. Run with `AITER_FLYDSL_MOE_COMPARE=1` for ONE bench iteration to confirm
   numerical parity with Triton reference.
3. Disable compare (`=0`), run the full benchmark.
4. Verify rocprof top-5 shows the new FlyDSL kernel, not fallback Triton.
5. Confirm BS=1 decode guard (`decode_bs1_in8k`) is within 2% of PR #23381.
6. Run `test_harness.py` — GSM8K must stay ≥ 0.90.

## 8. Common GEMM bugs

- **Acc dtype drift** — MFMA accumulates in fp32, but if you cast the C-fragment
  to bf16 before writing, you lose half the precision. Keep `rC` as f32 until
  the final store.
- **Wrong atom for arch** — compiled kernel for `MFMA_F32_32x32x8_BF16` (gfx942)
  launches on gfx950; runtime silently picks v1 atoms. Always gate atom
  selection on `fx.get_device_arch()`.
- **BLOCK_K too small for W4A16** — unpacking overhead dominates when
  `BLOCK_K < 128`. Use `BLOCK_K ≥ 128` for W4A16 unconditionally.
- **Bank conflicts in LDS** — see [flydsl-lds-optimization](../flydsl-lds-optimization/SKILL.md).
- **Split-K without atomic reduce** — your partial outputs don't sum. Use
  `fx.atomic_add` or `fx.split_k`.

## 9. Do NOT do

- Hand-write assembly to "beat FlyDSL". FlyDSL generates the same MFMA
  instructions CK does; the difference is usually layout, not ISA.
- Run FlyDSL and CK-based GEMMs in the same serving run — they fight for
  kernel cache space and each other's JIT slots.
- Modify `mixed_moe_gemm_2stage.py` without first running its internal
  test harness (`python3 -m kernels.mixed_moe_gemm_2stage --self-test`).
- Assume a kernel that beats Triton at M=16384 also beats at M=512. Decode
  shapes want different BLOCK_M + split-K.
