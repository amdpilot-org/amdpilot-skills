# AMD Suite Matrix

This is the practical runner/suite map for upstream SGLang AMD CI.

## PR / per-commit suites

From `.github/workflows/pr-test-amd.yml`:

| Suite | Runner label | When to use |
|---|---|---|
| `stage-a-test-1-gpu-small-amd` | `linux-mi325-1gpu-sglang` | Fast 1-GPU ROCm smoke / kernel correctness |
| `stage-b-test-1-gpu-small-amd` | `linux-mi325-1gpu-sglang` | Default 1-GPU AMD correctness |
| `stage-b-test-1-gpu-small-amd-nondeterministic` | `linux-mi325-1gpu-sglang` | Known nondeterministic 1-GPU AMD tests |
| `stage-b-test-1-gpu-small-amd-mi35x` | `linux-mi35x-gpu-1` | 1-GPU MI355X / gfx950 coverage |
| `stage-b-test-1-gpu-large-amd` | `linux-mi325-1gpu-sglang` | Heavy 1-GPU AMD tests |
| `stage-b-test-2-gpu-large-amd` | `linux-mi325-2gpu-sglang` | 2-GPU distributed AMD correctness |
| `stage-c-test-4-gpu-amd` | `linux-mi325-4gpu-sglang` | 4-GPU AMD scaling / DP / TP |
| `stage-c-test-large-8-gpu-amd` | `linux-mi325-8gpu-sglang` | 8-GPU MI325-scale validation |
| `stage-c-test-large-8-gpu-amd-mi35x` | `linux-mi35x-gpu-8` | 8-GPU MI355X / gfx950 validation |
| `stage-b-test-large-8-gpu-disaggregation-amd` | `linux-mi35x-gpu-8.fabric` | disaggregated AMD / fabric tests |

## Nightly suites

Nightly AMD suites live in `.github/workflows/nightly-test-amd-rocm720.yml`.

Common examples:

| Suite | Runner label |
|---|---|
| `nightly-amd-1-gpu` | `linux-mi325-1gpu-sglang` |
| `nightly-amd` | `linux-mi325-2gpu-sglang` |
| `nightly-amd-4-gpu` | `linux-mi325-4gpu-sglang` |
| `nightly-amd-1-gpu-mi35x` | `linux-mi35x-gpu-1` |
| `nightly-amd-8-gpu-mi35x` | `linux-mi35x-gpu-8` |

There are many model-specific nightly suites. When in doubt:

```bash
rg -n "run_suite.py --hw amd --suite|runs-on:" \
  .github/workflows/pr-test-amd.yml \
  .github/workflows/nightly-test-amd-rocm720.yml
```

## `register_amd_ci` patterns

### AMD-only test

```python
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=25, suite="stage-b-test-1-gpu-small-amd")
```

### Shared CUDA + AMD test

```python
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=100, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=100, suite="stage-b-test-1-gpu-small-amd")
```

Use this only when the same logical regression matters on both backends and the AMD path is worth the extra CI cost.

### MI35x/gfx950-specific coverage

```python
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=3600,
    suite="stage-c-test-large-8-gpu-amd-mi35x",
    disabled="move to nightly for saving time",
)
```

Use MI35x suites when the bug depends on gfx950 kernels, MI355X performance/correctness, or ROCm 7.2 image content.

## Local reproduction flow

### Standard AMD CI container

```bash
bash scripts/ci/amd/ensure_vram_clear.sh
bash scripts/ci/amd/amd_ci_start_container.sh --rocm-version rocm720
bash scripts/ci/amd/amd_ci_install_dependency.sh
bash scripts/ci/amd/amd_ci_exec.sh -w "/sglang-checkout/test" \
  python3 run_suite.py --hw amd --suite stage-b-test-1-gpu-small-amd
```

### MI35x-specific repro

```bash
bash scripts/ci/amd/ensure_vram_clear.sh
bash scripts/ci/amd/amd_ci_start_container.sh --rocm-version rocm720 --mi35x-base-tag v0.5.9-rocm720-mi35x
bash scripts/ci/amd/amd_ci_install_dependency.sh
bash scripts/ci/amd/amd_ci_exec.sh -w "/sglang-checkout/test" \
  python3 run_suite.py --hw amd --suite stage-b-test-1-gpu-small-amd-mi35x
```

### Direct file repro

```bash
bash scripts/ci/amd/amd_ci_exec.sh -w "/sglang-checkout/test" \
  python3 test/registered/quant/test_awq.py
```

## Important runner/image facts

- `amd_ci_start_container.sh` detects runner family from hostname.
- `mi30x|mi300|mi325` are normalized onto `mi30x` images.
- `mi35x` uses `mi35x` images.
- `amd_ci_exec.sh` injects:
  - `SGLANG_IS_IN_CI_AMD=1`
  - `SGLANG_IS_IN_CI=1`
  - `SGLANG_USE_AITER=1`
  - `GPU_ARCHS=gfx950` on MI35x only

That means MI35x bugs often reproduce only when you preserve the CI script path instead of running the test manually in a random container.
