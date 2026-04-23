---
name: flydsl-debug-kernel
description: >
  Diagnose FlyDSL kernel failures on AMD ROCm — JIT cache invalidation, gfx
  target mismatch (gfx942 vs gfx950), silent fallback, AITER_ENFORCE_DSL
  rationale, and mangled kernel names in rocprof traces. Use when a FlyDSL
  kernel imports but crashes, produces wrong numerics, or silently falls back
  to Triton, or when flydsl_moe_stage1 fails to dispatch.
---

# Debugging FlyDSL Kernels

**Golden rule:** run with `AITER_ENFORCE_DSL=1`. Without it, any import/compile
failure in the FlyDSL path falls back silently to Triton, and your "optimization
trial" will score whatever Triton scores — wasting the trial slot.

## 1. First debugging step: is FlyDSL actually in the hot path?

```bash
export AITER_ENFORCE_DSL=1
export AITER_USE_FLYDSL_MOE=1
export AITER_USE_FLYDSL_MOE_STAGE1=1
export AITER_USE_FLYDSL_MOE_STAGE2=1
# then run your benchmark
```

If the run errors with `RuntimeError: FlyDSL requested but unavailable …`,
the stack is broken — fix before measuring. If the run succeeds and you see
kernels named `fused_moe_kernel_gptq_awq` in `rocprof` output,
**FlyDSL never ran**. See §5 below.

## 2. JIT cache layout

FlyDSL compiles kernels on first invocation and caches them:

| Path | Purpose |
|---|---|
| `~/.flydsl/cache/<arch>/<hash>/` | Per-arch JIT cache (gfx942 or gfx950) |
| `/opt/FlyDSL/build-fly/python_packages/flydsl/_mlir*.so` | Compiler C-extension |
| `/opt/FlyDSL/build-fly/lib/` | FlyDSL runtime libs (must be on `LD_LIBRARY_PATH`) |

**Symptom:** a kernel that ran correctly in a previous session now dies with
`ValueError: arch mismatch`. **Cause:** container GPU changed arch since cache
was populated (e.g. migrated from MI300X gfx942 to MI355X gfx950). **Fix:**

```bash
rm -rf ~/.flydsl/cache
# or per-arch: rm -rf ~/.flydsl/cache/gfx942
```

To disable caching entirely for debugging (forces fresh compile every call):

```bash
export FLYDSL_RUNTIME_ENABLE_CACHE=0
```

## 3. gfx target mismatch symptoms

| What you see | What it means |
|---|---|
| `HSA Error: HSA_STATUS_ERROR_INVALID_ISA` | Cached kernel compiled for a different arch |
| `HIP error: invalid device function` | Same, from the HIP runtime side |
| `error: compiler did not emit a binary for gfx950` | FlyDSL's MLIR lowering didn't cover the target — upgrade to a newer main |
| CPU takes 30+ s on first call, then fast | Normal JIT behaviour; warm up before capturing CUDA graphs |

Detect the GPU arch without running the kernel:

```bash
rocminfo | grep -A2 "gfx" | head
# or programmatically:
/opt/venv/bin/python3 -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"
```

## 4. "wrong LLVM picked up" trap

Symptom at import: `undefined symbol: _ZSt3gcd…` or `undefined symbol: mlir::…`.

Root cause: the system `libMLIR.so` (from an apt-installed rocm-llvm-dev) loaded
before FlyDSL's own shipped MLIR. Fix the load order:

```bash
export LD_LIBRARY_PATH=/opt/FlyDSL/build-fly/lib:$MLIR_PATH/lib:$LD_LIBRARY_PATH
/opt/venv/bin/python3 -c "import flydsl._mlir"
```

If that still fails, `ldd /opt/FlyDSL/build-fly/python_packages/flydsl/_mlir*.so`
shows you which `libMLIR-*.so` is being resolved. It should be the one under
`/opt/FlyDSL/` or `$MLIR_PATH`, never `/usr/lib/`.

## 5. Silent-fallback detection

AITER dispatches FlyDSL MoE via try/except; if FlyDSL raises during
import OR during first call, AITER falls through to Triton. With
`AITER_ENFORCE_DSL=1` the exception becomes a hard error instead.

To confirm dispatch:

```python
import os; os.environ["AITER_ENFORCE_DSL"] = "1"
import aiter
from aiter.ops.fused_moe import flydsl_moe_stage1  # must import
# AITER_USE_FLYDSL_MOE* env vars control runtime dispatch inside this op
```

After a full benchmark run, `rocprof` top-kernel list should show FlyDSL's
kernels (MLIR-mangled names, see §7) — **not** `fused_moe_kernel_gptq_awq`.

## 6. Numerical-correctness debug

The blog ships two compare env vars that wrap every FlyDSL MoE call in a
reference Triton call and diff the output. Slow but exact.

```bash
export AITER_FLYDSL_MOE_COMPARE=1          # Stage 1 (gate/up)
export AITER_FLYDSL_MOE_COMPARE_STAGE2=1   # Stage 2 (down)
```

Enable ONLY during targeted debug — these add 2–3x overhead per call.
**Keep them OFF for any scored optimization trial** (they pollute throughput
numbers).

## 7. rocprof and mangled kernel names

FlyDSL-emitted kernels appear in `rocprof` with MLIR-mangled names such as
`flydsl_kernel_fused_moe_w4a16_stage1_kcall_0`. They do NOT show up as
plain `@flyc.kernel` Python names.

To correlate back to the Python source:

```bash
# Dump the MLIR that was compiled for a specific call:
FLYDSL_DUMP_IR=1 /opt/venv/bin/python3 your_test.py 2> mlir_dump.txt
# mlir_dump.txt now contains the full fly-dialect IR with source-location
# comments that point back to your @flyc.kernel function.
```

For serving traces, filter the `rocprof` JSON to just FlyDSL kernels:

```bash
rocprofv2 --plugin json sglang_server ... -- --kernel-filter 'flydsl_.*'
```

## 8. Checklist when a FlyDSL trial fails to improve

Before concluding "FlyDSL doesn't help on this shape":

1. Confirm `AITER_ENFORCE_DSL=1` and no exception was logged.
2. `grep flydsl /tmp/sglang_server.log` — look for "DSL stage1 hook active".
3. `rocprof` top-5 should show FlyDSL kernels (§7) not Triton `fused_moe_kernel`.
4. Check cache: `ls ~/.flydsl/cache/` is non-empty after the run.
5. JIT warm-up may count against the benchmark — run the bench twice,
   discard run 1. (Or set `FLYDSL_RUNTIME_ENABLE_CACHE=1` — default, should be.)
6. GPU arch check: `rocminfo | grep gfx9` — should be `gfx950` on MI355X.
7. `ldd /opt/FlyDSL/build-fly/python_packages/flydsl/_mlir*.so | grep -i mlir`
   → must resolve to FlyDSL's shipped MLIR, not `/usr/lib/…`.

If all of these pass and the metric still regresses, the kernel is genuinely
slower on this shape — see [flydsl-gemm-optimization](../flydsl-gemm-optimization/SKILL.md)
for kernel-level debugging.

## 9. Do NOT do

- Disable `AITER_ENFORCE_DSL` to "make the run succeed". A silent-fallback
  win is not a real win and will be rejected by the supervisor.
- Delete `/opt/FlyDSL/build-fly/` during a live run. The JIT links against
  symbols inside it.
- `pkill -f python` when the FlyDSL JIT is compiling — the child compiler
  processes matter. Use `kill <pid>` on the exact bench PID.
- Compare FlyDSL vs Triton kernels by wall-clock on run 1. First-call JIT
  is 5–30 s; kernel time only meaningful on steady-state.
