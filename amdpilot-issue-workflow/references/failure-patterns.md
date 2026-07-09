# AMDPilot Known Failure Patterns

Use this reference only when the main workflow points here. Keep entries evidence-shaped and promote repeated gotchas into durable system fixes.

## Worker SHA mismatch: pending despite free GPUs

`_required_worker_git_sha()` can default to the daemon's current git SHA. Restarting the daemon from a branch HEAD that differs from deployed remote workers can make every worker report `worker_status=stale: SHA mismatch`. Then tasks sit `status=pending` even when `/api/node/inventory` shows free GPUs.

Checks:

- Compare required worker SHA with each node's full `worker_metadata_json.git_sha` from node inventory storage; truncated UI SHAs are not enough.
- Confirm whether there are any eligible workers for the task's `gpu_arch`.
- If pinning, use the full 40-character SHA in `~/.config/amdpilot/serve.env`:

```bash
AMDPILOT_REQUIRED_WORKER_GIT_SHA=<40-char-sha-deployed-on-workers>
```

After a daemon restart, verify worker eligibility before launching another issue.

## Validator footgun: invalid baseline contracts can skip verification

If baseline-contract validation rejects the config, the gate may be skipped rather than failed. Do not “fix” a false pass by making `expected_metric` invalid.

Always verify these fields describe the same metric:

- `metric_pattern` extracts the line the harness actually prints.
- `metric_name` names that extracted value.
- `metric_direction` matches how higher/lower should be interpreted.
- `expected_metric` and tolerance form a meaningful threshold.

False-pass shape: a generic `SCORE: ...` regex paired with `metric_name=latency` and `direction=lower`, where a broken run prints `SCORE: 0.00` and incorrectly passes as zero latency.

For pass/fail-style harness scores where `0` is the expected broken-repo baseline and higher is improvement, prefer a coherent non-negative score contract such as `metric_direction: higher`, `expected_metric: 0.0`, `tolerance: 0.0`, but only when the verifier accepts and executes it. Do not copy this to throughput/latency tasks where nonzero expected performance is the point.

## False zombies from heartbeat or SQLite contention

`zombie_cleanup` can be a real dead container, but repeated false zombies can happen when the central daemon is delayed writing SQLite or responding to heartbeat routes long enough for GPU leases to look stale.

Before cleaning or relaunching, capture:

- `gpu_leases.heartbeat_at` and the configured stale threshold.
- Daemon logs around failure for SQLite lock storms or slow claim/upsert paths.
- Remote worker logs and process/container state on the assigned node.
- Whether live snapshots or agent steps were still advancing near cleanup time.

If confirmed false zombie, route to dispatcher/lease policy or daemon storage contention; do not patch eval artifacts to hide it. Increasing the stale threshold can reduce false reaping but trades off slower cleanup of truly dead containers.

## Heartbeats from `127.0.0.1` may be tunnelled remote workers

A daemon HTTP log such as `POST /api/queue/<id>/heartbeat 200 OK` from `127.0.0.1` does not prove the orchestrator is local. It may come through a tunnel from a remote-worker heartbeat thread. Check the assigned node's remote-worker log, container state, and `pgrep` there before concluding the orchestrator disappeared.

## `amdpilot-org/openpi#3`

`amdpilot-org/openpi@e4429ad` does not contain `benchmark_pi0_libero_rocm.py`. A valid artifact must fetch the canonical benchmark script from the issue-provided benchmark source before baseline verification. Include `bench_pi0_libero.sh` if `baseline_contract.reproduce_command` references it.

## `amdpilot-org/FlyDSL#2`

The harness prints a single `SCORE: 0..100` line. `100` means the checks passed; `0` means the kernel was not implemented. The `42.4 µs` number in the issue body is an inductor baseline reference, not the metric line the harness reports.

Canonical metric contract shape:

```yaml
baseline_contract:
  reproduce_command: /workspace/FlyDSL/.venv/bin/python /workspace/test_harness.py
  metric_pattern: 'SCORE:\s+([\d.]+)'
  metric_name: harness_checks_score
  metric_direction: higher
  expected_metric: 0.0
  tolerance: 0.0
```

Open follow-ups observed in prior runs:

1. The verifier may strip `PYTHONPATH` even when the harness requires `PYTHONPATH=/workspace/FlyDSL`. Either inline it into `baseline_contract.reproduce_command` or make the verifier propagate `task.yaml` container env.
2. If the score reaches partial credit but fails perf, inspect the kernel path rather than weakening the harness.
