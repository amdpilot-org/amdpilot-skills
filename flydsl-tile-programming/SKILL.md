---
name: flydsl-tile-programming
description: >
  FLIR (Flexible Layout IR) tile-programming primitives for FlyDSL kernels on
  AMD ROCm. Covers (Shape, Stride) layout algebra, make_shape / make_stride /
  make_layout / make_coord, crd2idx / idx2crd, composition / product / divide /
  raked_product, make_copy_atom / make_tiled_copy / get_slice / partition_S /
  partition_D. Use when you need to author or refactor a kernel using FlyDSL's
  layout algebra instead of raw indexing.
---

# FlyDSL Tile Programming (Layout Algebra)

FlyDSL's FLIR layout system is adapted from CuTe (NVIDIA CUTLASS's
layout algebra). If you're familiar with CuTe, map the concepts 1:1; otherwise
read §1 first.

## 1. Core concept: Layout = (Shape, Stride)

A **Layout** is a pair of nested integer tuples `(Shape, Stride)` that maps a
multi-dimensional coordinate to a linear offset:

```
Shape  = (M, N)
Stride = (1, M)       # column-major
```

Coordinate `(i, j)` → offset `i * 1 + j * M`.

Every tensor in FlyDSL carries a Layout. Unlike a raw stride, the Shape can be
nested: `((4, M/4), N)` means "think of the M axis as 4 outer tiles of M/4".

## 2. Primitive constructors

```python
from flydsl import fx

# Shape / Stride / Layout
S = fx.make_shape(M, N)                 # (M, N)
T = fx.make_stride(1, M)                # (1, M)
L = fx.make_layout(S, T)                # column-major layout

# Nested shapes for tile hierarchy:
S2 = fx.make_shape(fx.make_shape(4, M // 4), N)   # ((4, M/4), N)

# Coordinates
c = fx.make_coord(i, j)                 # (i, j)
offset = fx.crd2idx(c, L)               # linear index
c_back = fx.idx2crd(offset, L)          # reverse (coord from index)
```

## 3. The three composition operators

These are the most-used operators in real kernels — memorize them:

| Operator | Signature | What it does |
|---|---|---|
| `fx.composition(A, B)` | Layout × Layout → Layout | Apply B as a reshape of A. Most common for tile partitioning. |
| `fx.product(A, B)` | Layout × Layout → Layout | Cartesian product. Creates a tile-of-tiles layout. |
| `fx.divide(A, B)` | Layout × Layout → Layout | Inverse of product — split a layout into (tile, rest) |
| `fx.raked_product(A, B)` | Layout × Layout → Layout | Product with interleaved/striped ordering (common for warp-level tiling) |

Example — take an MxN tile, view it as (warps 0..3) x (per-warp tile):

```python
mn_layout = fx.make_layout(fx.make_shape(128, 128), fx.make_stride(1, 128))
warp_tiler = fx.make_layout(fx.make_shape(4, 1), fx.make_stride(1, 0))
warp_view = fx.divide(mn_layout, warp_tiler)
# warp_view is now a (warp, within_warp) layout
```

## 4. Copy atoms and tiled copies

FlyDSL expresses data movement (global → LDS → register) as **copy atoms**
parametrized by width and layout:

```python
# A "copy atom" is a single-instruction data-movement primitive
copy_atom = fx.make_copy_atom(
    atom=fx.UniversalCopy32b,     # 32-bit per-thread copy
    src_layout=global_layout,
    dst_layout=smem_layout,
)

# A "tiled copy" tiles the atom across a thread block
tiled_copy = fx.make_tiled_copy(
    copy_atom,
    thr_layout=fx.make_layout(fx.make_shape(16, 16)),  # 16x16 thread tile
    val_layout=fx.make_layout(fx.make_shape(4, 1)),    # 4 values per thread
)
```

Common copy atoms (AMD-specific):

| Atom | Purpose |
|---|---|
| `fx.UniversalCopy32b` | 32-bit per-lane (equivalent to 1 dword) |
| `fx.UniversalCopy128b` | 128-bit vectorized (4 dwords, one instruction) |
| `fx.AsyncCopyG2L` | Global-to-LDS async copy (overlaps with compute) |

## 5. Partition with get_slice

After building a tiled copy, partition source + destination for the current
thread:

```python
# Inside @flyc.kernel:
tid = fx.thread_idx.x
thr_copy = tiled_copy.get_slice(tid)

# Partition source (global) and destination (shared mem):
gS = thr_copy.partition_S(g_src)   # this thread's view of the source
gD = thr_copy.partition_D(s_dst)   # this thread's view of the destination

# Actually copy:
fx.copy(tiled_copy, gS, gD)
```

`partition_S` gives source slices (read-from). `partition_D` gives destination
slices (write-to). Always pair them; the tile shape must match between the two.

## 6. Typical kernel skeleton using layout algebra

```python
@flyc.kernel
def gemm_tile(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    M: fx.Constexpr[int], N: fx.Constexpr[int], K: fx.Constexpr[int],
    BLOCK_M: fx.Constexpr[int], BLOCK_N: fx.Constexpr[int], BLOCK_K: fx.Constexpr[int],
):
    bx, by = fx.block_idx.x, fx.block_idx.y
    tid = fx.thread_idx.x

    # 1. Build global layouts
    layA = fx.make_layout(fx.make_shape(M, K), fx.make_stride(K, 1))  # row-major
    layB = fx.make_layout(fx.make_shape(K, N), fx.make_stride(N, 1))
    layC = fx.make_layout(fx.make_shape(M, N), fx.make_stride(N, 1))

    # 2. Slice block tile
    gA = fx.local_tile(A, layA, (BLOCK_M, BLOCK_K), (bx, 0))
    gB = fx.local_tile(B, layB, (BLOCK_K, BLOCK_N), (0, by))
    gC = fx.local_tile(C, layC, (BLOCK_M, BLOCK_N), (bx, by))

    # 3. SMEM allocation + tiled copy
    sA = fx.smem_allocator.alloc(BLOCK_M, BLOCK_K, dtype=A.dtype)
    sB = fx.smem_allocator.alloc(BLOCK_K, BLOCK_N, dtype=B.dtype)
    tiled_copy_a = fx.make_tiled_copy(...)
    tiled_copy_b = fx.make_tiled_copy(...)

    # 4. Main K-loop
    for k in fx.range(0, K, BLOCK_K):
        fx.copy(tiled_copy_a.get_slice(tid), gA[:, k:k+BLOCK_K], sA)
        fx.copy(tiled_copy_b.get_slice(tid), gB[k:k+BLOCK_K, :], sB)
        fx.barrier()
        # ... MFMA compute on sA, sB (see flydsl-gemm-optimization)
        fx.barrier()

    # 5. Write back
    fx.copy(tiled_copy_c.get_slice(tid), sC, gC)
```

## 7. Common layout recipes

### Row-major MxN
```python
fx.make_layout(fx.make_shape(M, N), fx.make_stride(N, 1))
```

### Column-major MxN
```python
fx.make_layout(fx.make_shape(M, N), fx.make_stride(1, M))
```

### Swizzled LDS (for bank-conflict avoidance)
```python
swizzle = fx.make_swizzle(B=3, M=3, S=3)     # 8-bank swizzle
layS = fx.make_layout(fx.make_shape(M, N), fx.make_stride(1, M))
layS_swizzled = fx.composition(swizzle, layS)
```

See [flydsl-lds-optimization](../flydsl-lds-optimization/SKILL.md) for when to
use which swizzle.

### 4-warp partition over BLOCK_M
```python
warp_layout = fx.make_layout(fx.make_shape(4, 1), fx.make_stride(BLOCK_M // 4, 0))
warp_tile = fx.divide(block_layout, warp_layout)
```

## 8. Debugging layout bugs

1. Print offsets in Python (outside the kernel) — FlyDSL layouts are normal
   Python objects at trace time:

   ```python
   L = fx.make_layout(fx.make_shape(8, 8), fx.make_stride(1, 8))
   for i in range(8):
       for j in range(8):
           print(i, j, fx.crd2idx((i, j), L))
   ```

2. Use `fx.print_layout(L)` to dump a human-readable `(Shape, Stride)` form.
3. Shape-mismatch at launch: usually `partition_S` and `partition_D` disagree
   about tile size. Assert `gS.layout.shape == gD.layout.shape` before the
   copy.

## 9. Reference files in the container

```
/opt/FlyDSL/python/flydsl/layout/      # layout algebra implementation
/opt/FlyDSL/python/examples/01-vec-add.py
/opt/FlyDSL/python/examples/02-reduce.py
/opt/FlyDSL/python/examples/03-gemm.py
/opt/FlyDSL/python/examples/04-preshuffle_gemm.py   # start here for MoE GEMM
/opt/FlyDSL/python/kernels/moe_gemm_2stage.py       # K2.5 MoE kernel
```

Read `03-gemm.py` end-to-end before authoring a new GEMM — it's the shortest
path to understanding the full pipeline.

## 10. Do NOT do

- Hand-roll tile loops with manual indexing — layout algebra exists to
  eliminate those. If you catch yourself writing `a[i * stride_am + k * stride_ak]`
  you've regressed to pre-layout-algebra code.
- Mix row-major and column-major in the same Layout without a `composition` —
  FlyDSL won't warn; you'll just get garbage numerics.
- Allocate LDS outside `smem_allocator` — see [flydsl-lds-optimization](../flydsl-lds-optimization/SKILL.md).
- Call `fx.composition(A, B)` where `B.shape` is runtime-sized — the MLIR
  compiler needs static composition shapes.
