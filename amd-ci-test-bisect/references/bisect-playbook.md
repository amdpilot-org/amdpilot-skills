# AMD CI Regression Bisect Playbook

This is the AMD adaptation of the upstream `sglang-bisect-ci-regression` workflow.

## 1. Pick the right source of truth

AMD does **not** mirror NVIDIA's exact CI event model.

- Use `pr-test-amd.yml` on `main` for per-commit AMD regressions.
- Use `nightly-test-amd-rocm720.yml` with `event=schedule` for long / nightly AMD regressions.

### Mainline AMD pushes

```bash
gh run list \
  --repo sgl-project/sglang \
  --workflow "pr-test-amd.yml" \
  --branch main \
  --limit 20 \
  --json databaseId,conclusion,createdAt,headSha,event
```

### Scheduled AMD nightlies

```bash
gh run list \
  --repo sgl-project/sglang \
  --workflow "nightly-test-amd-rocm720.yml" \
  --event schedule \
  --branch main \
  --limit 20 \
  --json databaseId,conclusion,createdAt,headSha,event
```

## 2. Extract the failure signature

```bash
gh run view <RUN_ID> --repo sgl-project/sglang --json jobs \
  --jq '.jobs[] | select(.conclusion == "failure") | {name, conclusion, databaseId}'

gh run view <RUN_ID> --repo sgl-project/sglang --job <JOB_ID> --log 2>&1 | \
  grep -E -B 5 -A 40 "AssertionError|FAIL|Error|RuntimeError|rocm|RCCL|HIP|gfx950|mi35x"
```

Capture:

- exact failing test file and method
- assertion or crash string
- runner name / hardware label
- ROCm version and image tag if visible

## 3. Always classify the runner first

Before blaming a commit, extract:

```bash
gh run view <RUN_ID> --repo sgl-project/sglang --job <JOB_ID> --log 2>&1 | \
  grep -E "Runner name|Machine name|Hostname|rocm/sgl-dev|ROCm|gfx950|gfx942|AITER_COMMIT|Driver Version"
```

For AMD, the most important drift dimensions are:

1. runner family: `linux-mi325-*` vs `linux-mi35x-*`
2. GPU arch: `gfx942` vs `gfx950`
3. ROCm version / image tag: `rocm700` vs `rocm720`, date-stamped image changes

If the same SHA passes on MI325 and fails on MI35x, classify it as hardware/image-specific first, not a generic code regression.

## 4. Find the pass/fail boundary

Once you know whether this is a push-run or nightly issue, find:

- last passing run on the same suite family
- first failing run on the same suite family

Then list commits:

```bash
git log --oneline <LAST_PASS_SHA>..<FIRST_FAIL_SHA>
```

If the run family changed runner/image between the two points, do **not** call it a clean code bisect yet.

## 5. Reproduce with the AMD CI scripts

The shortest faithful reproduction is:

```bash
bash scripts/ci/amd/ensure_vram_clear.sh
bash scripts/ci/amd/amd_ci_start_container.sh --rocm-version rocm720
bash scripts/ci/amd/amd_ci_install_dependency.sh
bash scripts/ci/amd/amd_ci_exec.sh -w "/sglang-checkout/test" \
  python3 run_suite.py --hw amd --suite <SUITE_NAME>
```

If the failure is MI35x-specific, reproduce on an MI35x runner or keep the MI35x image tag and `GPU_ARCHS=gfx950` path intact.

## 6. Classification rubric

| Pattern | Diagnosis |
|---|---|
| Same SHA fails on MI35x, passes on MI325 | gfx950 / MI355X-specific |
| Same suite starts failing after a single code range on identical runner family | code regression |
| Failures begin after ROCm/image tag change with no clear code boundary | environment/image drift |
| Same runner flips pass/fail on same SHA | flaky / nondeterministic |
| TP=1 passes, multi-GPU AMD fails | RCCL / distributed AMD issue |

## 7. Required report shape

```markdown
## AMD CI Regression Report

- Test:
- Workflow:
- Suite / runner:
- GPU family:
- ROCm / image tag:
- Last pass:
- First fail:
- Classification:
- Evidence:
- Local reproduction command:
- Recommended fix:
```

Do not omit the suite / runner / GPU family fields. On AMD they are often the root cause.
