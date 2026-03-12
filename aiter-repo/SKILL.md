---
name: aiter-repo
description: >
  Common pitfalls and practical tips for working on the AITER codebase.
  Loaded automatically when the target repo is ROCm/aiter.
---

# AITER: Common Pitfalls

## 1. JIT modules are NOT rebuilt by `setup.py develop`

AITER compiles kernels lazily (JIT on first call). Pre-compiled `.so` files
live in `aiter/jit/`.  After editing a `.cu` source file, **you must delete
the cached module** or your changes will not take effect:

```bash
rm -rf aiter/jit/<module_name>.so aiter/jit/build/<module_name>
```

Then run your test — the JIT system will rebuild automatically.
The module-to-source mapping is in `aiter/jit/optCompilerConfig.json`.

## 2. Template-compiled kernels use a separate cache

Kernels built via `compile_template_op` (e.g. sampling) cache in
`/root/.aiter/build/<hash>/`, not in `aiter/jit/`.  Delete the matching
hash directory after editing `.cuh` headers.

## 3. AMD hardware intrinsics for fast math

MI300X/MI350X/MI355X expose single-instruction intrinsics that are
significantly faster than standard math:

```cpp
__builtin_amdgcn_exp2f(x)    // fast exp2
__builtin_amdgcn_rcpf(x)     // fast reciprocal
__builtin_amdgcn_log2f(x)    // fast log2
__builtin_amdgcn_rsqf(x)     // fast rsqrt
```

Convert `exp(x)` → `exp2(x * LOG2E)` + `rcpf` for sigmoid.
Search for existing usage: `grep -rn "__builtin_amdgcn" csrc/`.

## 4. `test_mla.py` uses argparse — run it directly

`op_tests/test_mla.py` has its own argparse that conflicts with pytest.
Run it as a standalone script, not through pytest:

```bash
python3 op_tests/test_mla.py -n 16,1 -b 1 -c 4000
```

## 5. MLA persistent vs non-persistent code paths

`mla_decode_fwd` has two reduce paths.  The non-persistent path (default in
tests) uses a Triton stage-2 kernel; the persistent path (used in production)
calls `mla_reduce_v1` from `module_mla_reduce`.  If your task targets
`reduce.cu`, make sure you exercise the persistent path.
