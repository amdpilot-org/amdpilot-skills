---
name: flydsl-build
description: >
  Build, install, and verify the FlyDSL stack inside an AMD ROCm container.
  Covers scripts/build.sh, MLIR_PATH auto-detection, PYTHONPATH/LD_LIBRARY_PATH
  fallback when pip editable isn't available, and ROCm 7.x compatibility. Use
  when FlyDSL import fails, build errors mention MLIR, or you need to verify
  DSL2_ROOT / flyc.kernel is loadable.
---

# Building FlyDSL on ROCm

The K2.6 eval image ships FlyDSL prebuilt at `/opt/FlyDSL`. Read this skill
only if (a) the image was built without FlyDSL, (b) you need to rebuild after
editing FlyDSL internals, or (c) `import flydsl` fails at runtime.

## 1. First check: is it already installed?

```bash
/opt/venv/bin/python3 -c "import flydsl; print(flydsl.__file__)"
ls -la /opt/FlyDSL/python/flydsl/
echo "$DSL2_ROOT $MLIR_PATH"
```

Expected: `flydsl.__file__` under `/opt/FlyDSL/python/flydsl/__init__.py`,
`DSL2_ROOT=/opt/FlyDSL`, `MLIR_PATH=/opt/mlir_install`.

If those print correctly, **skip the rest of this skill** — you're already
set up. Go to [flydsl-debug-kernel](../flydsl-debug-kernel/SKILL.md) for
runtime import errors.

## 2. From-source build (cold cache)

```bash
cd /opt && git clone --depth 1 https://github.com/ROCm/FlyDSL.git
cd FlyDSL

bash scripts/build_llvm.sh -j64
bash scripts/build.sh -j64

/opt/venv/bin/pip install -e python/
```

`build_llvm.sh` compiles the custom MLIR dependency into `build-llvm/install/`.
If `MLIR_PATH` is unset the FlyDSL build step auto-detects this path.

`build.sh` produces `build-fly/python_packages/flydsl/` which is what
`pip install -e python/` points at. Expect 20–40 minutes cold, 2–5 minutes
incremental.

## 3. Prebuilt MLIR fallback

If `build_llvm.sh` fails or is too slow, try the ROCm apt packages as MLIR
source:

```bash
apt install -y rocm-llvm-dev rocm-llvm-tools
export MLIR_PATH=/opt/rocm/llvm
bash scripts/build.sh -j64
```

Not all FlyDSL passes are compatible with every ROCm LLVM version — if the
build errors on `std::gcd not found` or similar C++17 mismatches, you have a
toolchain split. Rebuild MLIR from `scripts/build_llvm.sh` instead.

## 4. PYTHONPATH / LD_LIBRARY_PATH fallback

When `/opt/venv/bin/pip install -e python/` fails (read-only venv, permissions,
out-of-sync resolver), point Python at the build output directly:

```bash
export PYTHONPATH=/opt/FlyDSL/python:/opt/FlyDSL/build-fly/python_packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/FlyDSL/build-fly/lib:$MLIR_PATH/lib:$LD_LIBRARY_PATH
/opt/venv/bin/python3 -c "import flydsl, flydsl._mlir"
```

`flydsl._mlir` is the C-extension that dies with `ImportError: undefined
symbol: …` when `LD_LIBRARY_PATH` is wrong. Both the `flydsl/` Python package
AND the built `_mlir*.so` under `build-fly/python_packages/flydsl/` need to
be on the path.

## 5. ROCm version compatibility

| ROCm | FlyDSL main | Notes |
|---|---|---|
| 7.2.0 | works | Verified by Kimi-K2.5 blog + K2.6 eval image |
| 7.0.x | works with rebuild | Rebuild `build_llvm.sh` — shipped MLIR may be 7.2-linked |
| 6.x | fails | Missing ROCDL intrinsics the FlyDSL compiler emits for gfx95x |

The K2.6 eval base image (`jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420`)
is pinned to ROCm 7.2.0. Do NOT bump ROCm without rebuilding FlyDSL.

## 6. Kernel self-test

After import works, verify a trivial kernel runs end-to-end on the actual GPU:

```python
import flydsl
import flydsl.flyc as flyc
from flydsl import fx
import torch

@flyc.kernel
def vec_add(a: fx.Tensor, b: fx.Tensor, c: fx.Tensor, n: fx.Constexpr[int]):
    i = fx.block_idx.x * 256 + fx.thread_idx.x
    if i < n:
        c[i] = a[i] + b[i]

N = 1024
a = torch.randn(N, device="cuda")
b = torch.randn(N, device="cuda")
c = torch.empty_like(a)
vec_add.launch(grid=((N + 255) // 256, 1, 1), block=(256, 1, 1))(a, b, c, N)
torch.testing.assert_close(c, a + b)
print("FlyDSL end-to-end OK")
```

If this fails, drop into [flydsl-debug-kernel](../flydsl-debug-kernel/SKILL.md).
If it passes, you can trust the install for MoE / GEMM work.

## 7. AITER integration check (K2.6 path)

AITER's `dev/kimi-K2.5` branch dispatches MoE through FlyDSL when
`DSL2_ROOT` is on `sys.path` and `AITER_USE_FLYDSL_MOE=1`. Verify the hook:

```bash
cd /sgl-workspace/aiter
git branch --show-current                      # must be dev/kimi-K2.5
/opt/venv/bin/python3 -c "
import sys; sys.path.insert(0, '/opt/FlyDSL/python')
from aiter.ops.fused_moe import flydsl_moe_stage1  # must import without error
print('AITER FlyDSL hook OK')
"
```

If import fails with `ImportError: cannot import name 'flydsl_moe_stage1'`,
AITER is on the wrong branch. The K2.5 MoE kernels live ONLY on
`dev/kimi-K2.5`.

## 8. Common build failures

- `CMake Error … MLIRConfig.cmake not found` → `MLIR_PATH` unset or points at
  a non-install dir. Check `ls $MLIR_PATH/lib/cmake/mlir/MLIRConfig.cmake`.
- `error: std::gcd not found` → wrong LLVM version picked up. Unset
  `MLIR_PATH`, let `build.sh` find its own MLIR via `build-llvm/install/`.
- `error: no member named 'CallInterfaceCallable'` → MLIR too new/old for
  this FlyDSL commit. Git bisect FlyDSL main if this is a blocker; usually
  means FlyDSL main has moved ahead of the container's ROCm MLIR snapshot.
- `pip install -e python/` exits 0 but `import flydsl` says `ModuleNotFoundError`
  → pip installed the stub but built artifacts are missing. Check
  `ls /opt/FlyDSL/build-fly/python_packages/flydsl/_mlir*.so` — if empty,
  rerun `bash scripts/build.sh -j64` and watch for errors.

## 9. Do NOT do

- `pip install -e .` on `/opt/FlyDSL` (top-level) — FlyDSL's setup lives in
  `python/setup.py`. The top-level `pyproject.toml` is a shim.
- Delete `build-fly/` between runs on the same machine — that's the kernel
  JIT cache. Clearing it is a 30-minute penalty on next launch.
- Export `CUDA_VISIBLE_DEVICES` to restrict GPUs — FlyDSL respects
  `HIP_VISIBLE_DEVICES`. `CUDA_*` env vars get translated but can cause
  double-filtering.
