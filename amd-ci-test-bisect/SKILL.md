---
name: amd-ci-test-bisect
description: >
  Add or update AMD/ROCm SGLang regression tests, choose the right MI325/MI35x
  CI suite, reproduce AMD CI locally with the upstream amd_ci container scripts,
  and bisect AMD CI regressions on main/nightly. Use when an agent-generated fix
  needs register_amd_ci coverage, when selecting MI325 vs MI35x runners, when
  debugging pr-test-amd or nightly AMD failures, or when documenting the
  register_amd_ci flow for ROCm-specific fixes.
---

# AMD CI Test / Bisect

This skill is the AMD/ROCm adaptation of SGLang's `write-sglang-test`,
`ci-workflow-guide`, and `sglang-bisect-ci-regression` skills.

Use it for four things:

1. add a regression test for an AMD-only or ROCm-sensitive fix
2. register that test with `register_amd_ci(...)`
3. reproduce the AMD CI job locally through the upstream `scripts/ci/amd/*` flow
4. bisect an AMD CI regression across MI325 / MI35x runners, ROCm versions, and image tags

Read [references/suite-matrix.md](references/suite-matrix.md) when you need exact suite/runner mappings or local reproduction commands.

Read [references/bisect-playbook.md](references/bisect-playbook.md) when you are debugging a failing AMD CI run on `main` or nightly.

## Core Rules

1. **`register_amd_ci` is AST-parsed.**
   Keep `est_time`, `suite`, `nightly`, and `disabled` as module-level literal constants.
   Do not hide them behind helper functions, variables, or computed expressions.

2. **Only add AMD registration when AMD coverage is the point.**
   Use `register_amd_ci(...)` for ROCm-only kernels, HIP/aiter paths, MI35x/gfx950 behavior,
   RCCL/distributed AMD paths, or an AMD regression fix.
   Do not duplicate backend-independent tests onto AMD just because AMD exists.

3. **Choose the lightest AMD suite that proves the fix.**
   Prefer 1-GPU MI325 first.
   Escalate to MI35x only when the failure depends on gfx950 / MI355X / ROCm 7.2 image behavior.
   Escalate to 2/4/8 GPU only when the bug requires distributed state, DP/TP, or model scale.

4. **MI325 and MI35x are different validation targets.**
   In upstream AMD CI, MI325-class runners are generally normalized onto `mi30x` images.
   MI35x runners use MI35x images and `amd_ci_exec.sh` injects `GPU_ARCHS=gfx950`.
   Treat them as separate contracts, not interchangeable hardware.

5. **Reproduce with the AMD CI scripts, not ad-hoc shell state.**
   Use `ensure_vram_clear.sh`, `amd_ci_start_container.sh`, `amd_ci_install_dependency.sh`,
   and `amd_ci_exec.sh`. That is the closest match to what GitHub Actions actually runs.

6. **For regressions, separate code regressions from runner/image drift.**
   Always record runner label, GPU family, ROCm version, and container image tag
   before blaming a commit.

## Workflow

### Phase 1: Classify the fix

Ask:

- Is the bug AMD-only, or just observed first on AMD?
- Does it depend on `is_in_amd_ci()`, HIP kernels, aiter, RCCL, or ROCm image contents?
- Is it MI35x/gfx950-specific, or should MI325 coverage be enough?
- Does it need 1 GPU, 2 GPU, 4 GPU, or 8 GPU to reproduce?

If the answer is "common logic, no AMD-specific path", keep the test on CPU/CUDA and do not add AMD registration just for symmetry.

### Phase 2: Add the test

Default SGLang authoring rules still apply:

- use `CustomTestCase`
- make `tearDownClass` defensive
- prefer mocks/unit tests when a server is unnecessary
- place CI-discovered tests under `test/registered/**`

For AMD-specific behavior, the common pattern is:

```python
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")
```

When the same file should run on both CUDA and AMD, register both explicitly and gate behavior inside the test body with `is_in_amd_ci()`.

Use `disabled="reason"` instead of deleting coverage when a suite is temporarily too expensive or unstable.

### Phase 3: Pick the suite

Use [references/suite-matrix.md](references/suite-matrix.md).

Practical default:

- 1-GPU ROCm correctness/kernel issue: `stage-a-test-1-gpu-small-amd` or `stage-b-test-1-gpu-small-amd`
- 1-GPU heavy model / memory issue on AMD: `stage-b-test-1-gpu-large-amd`
- MI355X/gfx950-specific issue: `stage-b-test-1-gpu-small-amd-mi35x` or `stage-c-test-large-8-gpu-amd-mi35x`
- distributed AMD issue: `stage-b-test-2-gpu-large-amd` or `stage-c-test-4-gpu-amd`
- long model/eval coverage: nightly AMD suites

### Phase 4: Reproduce locally

From a SGLang checkout:

```bash
bash scripts/ci/amd/ensure_vram_clear.sh
bash scripts/ci/amd/amd_ci_start_container.sh --rocm-version rocm720
bash scripts/ci/amd/amd_ci_install_dependency.sh
bash scripts/ci/amd/amd_ci_exec.sh -w "/sglang-checkout/test" \
  python3 run_suite.py --hw amd --suite stage-b-test-1-gpu-small-amd
```

Notes:

- `amd_ci_start_container.sh` auto-detects MI325/MI35x from the runner hostname.
- MI325/MI300 runners collapse to `mi30x` images; MI35x keeps `mi35x`.
- `amd_ci_exec.sh` auto-adds `SGLANG_IS_IN_CI_AMD=1`, `SGLANG_USE_AITER=1`,
  and on MI35x also `GPU_ARCHS=gfx950`.
- For disaggregated tests, use `amd_ci_start_container_disagg.sh`.

### Phase 5: Bisect regressions

Use [references/bisect-playbook.md](references/bisect-playbook.md).

The AMD twist is important:

- `pr-test-amd.yml` does **push / PR / rerun-stage** coverage, not scheduled cron
- `nightly-test-amd-rocm720.yml` is the scheduled source of truth for long AMD coverage

So:

- use **push runs on `main`** for per-commit AMD regressions
- use **scheduled nightly AMD runs** for large-model / MI35x / long-running regressions

## Output Requirements

When you finish an AMD test / CI / bisect task, report:

1. whether the fix is AMD-only or cross-backend
2. the chosen `register_amd_ci` suite and why
3. whether validation target is MI325/mi30x or MI35x/gfx950
4. the exact local reproduction command
5. if bisecting, whether the root cause is:
   - code regression
   - runner/hardware-specific
   - image / ROCm version drift
   - flaky / nondeterministic

## Minimal Checklist

- test added or updated in the right folder
- `register_amd_ci(...)` literals are valid for AST parsing
- suite matches the real hardware requirement
- local AMD CI reproduction command was documented
- any MI35x-only assumption is stated explicitly
- reviewer can tell whether this should also keep CUDA coverage
